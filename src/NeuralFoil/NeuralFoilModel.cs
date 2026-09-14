namespace NeuralFoil;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;

/// <summary>
/// A pure C# port of NeuralFoil's inference core (<c>get_aero_from_kulfan_parameters</c>). Evaluates the
/// trained multilayer perceptron for a given airfoil (expressed in Kulfan/CST parameters) and set of flow
/// conditions, applying the same Mahalanobis-based confidence, alpha-symmetry fusion, and output decoding
/// as the reference implementation. Model weights are read directly from the distributed <c>.npz</c> files.
/// </summary>
public static class NeuralFoilModel
{
    /// <summary>Number of boundary-layer evaluation stations per surface.</summary>
    public const int N = 32;

    private const int InputCount = 25;
    private const double Eps = 10.0 / double.MaxValue;
    private static readonly double LnEps = Math.Log(Eps);

    /// <summary>
    /// The x/c stations (per surface) at which boundary-layer quantities are reported, matching
    /// NeuralFoil's <c>bl_x_points</c>.
    /// </summary>
    public static double[] BlXPoints { get; } = ComputeBlXPoints(N);

    /// <summary>The eight model sizes NeuralFoil ships, from least to most accurate/expensive.</summary>
    public static IReadOnlyList<string> ModelSizes { get; } =
    [
        "xxsmall", "xsmall", "small", "medium", "large", "xlarge", "xxlarge", "xxxlarge",
    ];

    /// <summary>Evaluate a single case. See the array overload for full parameter documentation.</summary>
    public static AeroResults GetAeroFromKulfanParameters(
        KulfanParameters kulfan,
        double alpha,
        double re,
        double nCrit = 9.0,
        double xtrUpper = 1.0,
        double xtrLower = 1.0,
        string modelSize = "xlarge",
        string? weightsDirectory = null)
        => GetAeroFromKulfanParameters(
            kulfan, [alpha], [re], [nCrit], [xtrUpper], [xtrLower], modelSize, weightsDirectory);

    /// <summary>
    /// Evaluate a batch of cases for a single airfoil. The scalar flow inputs (<paramref name="alpha"/>,
    /// <paramref name="re"/>, <paramref name="nCrit"/>, <paramref name="xtrUpper"/>, <paramref name="xtrLower"/>)
    /// are broadcast NumPy-style: each may have length 1 or the common case count.
    /// </summary>
    /// <param name="kulfan">The airfoil, as 8-per-side Kulfan (CST) parameters.</param>
    /// <param name="alpha">Angle(s) of attack, in degrees.</param>
    /// <param name="re">Reynolds number(s).</param>
    /// <param name="nCrit">Critical amplification factor(s) for transition. Defaults to 9.</param>
    /// <param name="xtrUpper">Forced upper-surface transition x/c. Defaults to 1 (natural).</param>
    /// <param name="xtrLower">Forced lower-surface transition x/c. Defaults to 1 (natural).</param>
    /// <param name="modelSize">One of <see cref="ModelSizes"/>. Defaults to "xlarge".</param>
    /// <param name="weightsDirectory">
    /// Optional explicit path to the <c>nn_weights_and_biases</c> directory. When null, the directory is
    /// resolved automatically relative to the loaded assembly and working directory.
    /// </param>
    public static AeroResults GetAeroFromKulfanParameters(
        KulfanParameters kulfan,
        double[] alpha,
        double[] re,
        double[]? nCrit = null,
        double[]? xtrUpper = null,
        double[]? xtrLower = null,
        string modelSize = "xlarge",
        string? weightsDirectory = null)
    {
        if (kulfan is null) throw new ArgumentNullException(nameof(kulfan));
        if (alpha is null) throw new ArgumentNullException(nameof(alpha));
        if (re is null) throw new ArgumentNullException(nameof(re));
        kulfan.Validate();

        nCrit ??= [9.0];
        xtrUpper ??= [1.0];
        xtrLower ??= [1.0];

        if (!ModelSizes.Contains(modelSize))
            throw new ArgumentException(
                $"Invalid model_size '{modelSize}'. Must be one of: {string.Join(", ", ModelSizes)}.");

        var repository = weightsDirectory is null
            ? WeightRepository.Default
            : new WeightRepository(weightsDirectory);
        var network = repository.GetNetwork(modelSize);
        var distribution = repository.GetDistribution();

        var nCases = DetermineCaseCount(alpha, re, nCrit, xtrUpper, xtrLower);

        var alphaB = Broadcast(alpha, nCases, nameof(alpha));
        var reB = Broadcast(re, nCases, nameof(re));
        var nCritB = Broadcast(nCrit, nCases, nameof(nCrit));
        var xtrUpperB = Broadcast(xtrUpper, nCases, nameof(xtrUpper));
        var xtrLowerB = Broadcast(xtrLower, nCases, nameof(xtrLower));

        var x = BuildInputs(kulfan, alphaB, reB, nCritB, xtrUpperB, xtrLowerB, nCases);

        // Direct evaluation.
        var y = Net(network, x);
        SubtractConfidencePenalty(y, x, distribution);

        // Symmetry-embedded evaluation: flip inputs across alpha, re-run, then unflip the outputs.
        var xFlipped = FlipInputs(x);
        var yFlipped = Net(network, xFlipped);
        SubtractConfidencePenalty(yFlipped, xFlipped, distribution);
        var yUnflipped = UnflipOutputs(yFlipped);

        // Fuse the two evaluations and apply final activations.
        var yFused = (y + yUnflipped) / 2.0;
        return DecodeOutputs(yFused, reB, nCases);
    }

    private static Matrix<double> BuildInputs(
        KulfanParameters kulfan,
        double[] alpha,
        double[] re,
        double[] nCrit,
        double[] xtrUpper,
        double[] xtrLower,
        int nCases)
    {
        var x = new DenseMatrix(nCases, InputCount);
        for (var c = 0; c < nCases; c++)
        {
            var a = alpha[c] * Math.PI / 180.0; // degrees -> radians for trig
            for (var i = 0; i < 8; i++)
            {
                x[c, i] = kulfan.UpperWeights[i];
                x[c, 8 + i] = kulfan.LowerWeights[i];
            }
            x[c, 16] = kulfan.LeadingEdgeWeight;
            x[c, 17] = kulfan.TrailingEdgeThickness * 50.0;
            x[c, 18] = Math.Sin(2.0 * a);
            x[c, 19] = Math.Cos(a);
            x[c, 20] = 1.0 - Math.Cos(a) * Math.Cos(a);
            x[c, 21] = (Math.Log(re[c]) - 12.5) / 3.5;
            x[c, 22] = (nCrit[c] - 9.0) / 4.5;
            x[c, 23] = xtrUpper[c];
            x[c, 24] = xtrLower[c];
        }
        return x;
    }

    private static Matrix<double> Net(NeuralNetwork network, Matrix<double> x)
    {
        // Work in (features, cases) orientation, matching the reference implementation.
        var h = x.Transpose();
        var layers = network.Layers;
        for (var li = 0; li < layers.Count; li++)
        {
            h = layers[li].Weight * h;
            AddBiasInPlace(h, layers[li].Bias);
            if (li < layers.Count - 1)
                SwishInPlace(h);
        }
        return h.Transpose();
    }

    private static void AddBiasInPlace(Matrix<double> h, Vector<double> bias)
    {
        for (var r = 0; r < h.RowCount; r++)
        {
            var b = bias[r];
            for (var col = 0; col < h.ColumnCount; col++)
                h[r, col] += b;
        }
    }

    private static void SwishInPlace(Matrix<double> h)
    {
        // SiLU / Swish: x * sigmoid(x) = x / (1 + exp(-x)).
        for (var r = 0; r < h.RowCount; r++)
            for (var c = 0; c < h.ColumnCount; c++)
            {
                var v = h[r, c];
                h[r, c] = v / (1.0 + Math.Exp(-v));
            }
    }

    private static void SubtractConfidencePenalty(
        Matrix<double> y, Matrix<double> x, InputDistribution distribution)
    {
        // Baked into training so confidence decays to zero far from the training data.
        var diff = x.Clone();
        for (var r = 0; r < diff.RowCount; r++)
            for (var c = 0; c < diff.ColumnCount; c++)
                diff[r, c] -= distribution.Mean[c];

        var weighted = diff * distribution.InverseCovariance;
        var denominator = 2.0 * distribution.InputCount;
        for (var r = 0; r < y.RowCount; r++)
        {
            var sq = 0.0;
            for (var c = 0; c < diff.ColumnCount; c++)
                sq += weighted[r, c] * diff[r, c];
            y[r, 0] -= sq / denominator;
        }
    }

    private static Matrix<double> FlipInputs(Matrix<double> x)
    {
        var f = x.Clone();
        for (var r = 0; r < x.RowCount; r++)
        {
            for (var i = 0; i < 8; i++)
            {
                f[r, i] = x[r, 8 + i] * -1.0;      // upper <- flipped lower
                f[r, 8 + i] = x[r, i] * -1.0;       // lower <- flipped upper
            }
            f[r, 16] = -x[r, 16];                    // leading-edge weight
            f[r, 18] = -x[r, 18];                    // sin(2a)
            f[r, 23] = x[r, 24];                     // swap xtr_upper / xtr_lower
            f[r, 24] = x[r, 23];
        }
        return f;
    }

    private static Matrix<double> UnflipOutputs(Matrix<double> yFlipped)
    {
        var u = yFlipped.Clone();
        for (var r = 0; r < yFlipped.RowCount; r++)
        {
            u[r, 1] = yFlipped[r, 1] * -1.0; // CL
            u[r, 3] = yFlipped[r, 3] * -1.0; // CM
            u[r, 4] = yFlipped[r, 5];        // Top_Xtr <- Bot_Xtr
            u[r, 5] = yFlipped[r, 4];        // Bot_Xtr <- Top_Xtr

            for (var i = 0; i < 2 * N; i++)
            {
                // Swap upper (theta, H) block with lower (theta, H) block.
                u[r, 6 + i] = yFlipped[r, 6 + 3 * N + i];
                u[r, 6 + 3 * N + i] = yFlipped[r, 6 + i];
            }
            for (var i = 0; i < N; i++)
            {
                // Swap and negate upper/lower ue/vInf blocks.
                u[r, 6 + 2 * N + i] = -1.0 * yFlipped[r, 6 + 5 * N + i];
                u[r, 6 + 5 * N + i] = -1.0 * yFlipped[r, 6 + 2 * N + i];
            }
        }
        return u;
    }

    private static AeroResults DecodeOutputs(Matrix<double> y, double[] re, int nCases)
    {
        var confidence = new double[nCases];
        var cl = new double[nCases];
        var cd = new double[nCases];
        var cm = new double[nCases];
        var topXtr = new double[nCases];
        var botXtr = new double[nCases];
        var upperTheta = new double[nCases][];
        var upperH = new double[nCases][];
        var upperUe = new double[nCases][];
        var lowerTheta = new double[nCases][];
        var lowerH = new double[nCases][];
        var lowerUe = new double[nCases][];

        for (var r = 0; r < nCases; r++)
        {
            confidence[r] = Sigmoid(y[r, 0]);
            cl[r] = y[r, 1] / 2.0;
            cd[r] = Math.Exp((y[r, 2] - 2.0) * 2.0);
            cm[r] = y[r, 3] / 20.0;
            topXtr[r] = Math.Clamp(y[r, 4], 0.0, 1.0);
            botXtr[r] = Math.Clamp(y[r, 5], 0.0, 1.0);

            var uUe = new double[N];
            var lUe = new double[N];
            var uTheta = new double[N];
            var uH = new double[N];
            var lTheta = new double[N];
            var lH = new double[N];

            for (var i = 0; i < N; i++)
            {
                uUe[i] = y[r, 6 + 2 * N + i];
                lUe[i] = y[r, 6 + 5 * N + i];

                uTheta[i] = (Math.Pow(10.0, y[r, 6 + i]) - 0.1) / (Math.Abs(uUe[i]) * re[r]);
                uH[i] = 2.6 * Math.Exp(y[r, 6 + N + i]);
                lTheta[i] = (Math.Pow(10.0, y[r, 6 + 3 * N + i]) - 0.1) / (Math.Abs(lUe[i]) * re[r]);
                lH[i] = 2.6 * Math.Exp(y[r, 6 + 4 * N + i]);
            }

            upperUe[r] = uUe;
            lowerUe[r] = lUe;
            upperTheta[r] = uTheta;
            upperH[r] = uH;
            lowerTheta[r] = lTheta;
            lowerH[r] = lH;
        }

        return new AeroResults(
            confidence, cl, cd, cm, topXtr, botXtr,
            upperTheta, upperH, upperUe,
            lowerTheta, lowerH, lowerUe);
    }

    private static double Sigmoid(double x)
    {
        x = Math.Clamp(x, LnEps, -LnEps);
        return 1.0 / (1.0 + Math.Exp(-x));
    }

    private static int DetermineCaseCount(params double[][] inputs)
    {
        var n = 1;
        foreach (var input in inputs)
        {
            if (input.Length > 1)
            {
                if (n == 1)
                    n = input.Length;
                else if (input.Length != n)
                    throw new ArgumentException(
                        $"All array inputs must have the same length. Conflicting lengths: {n} and {input.Length}.");
            }
        }
        return n;
    }

    private static double[] Broadcast(double[] input, int nCases, string name)
    {
        if (input.Length == nCases)
            return input;
        if (input.Length == 1)
        {
            var result = new double[nCases];
            Array.Fill(result, input[0]);
            return result;
        }
        throw new ArgumentException(
            $"Input '{name}' has length {input.Length}, which cannot broadcast to {nCases} cases.");
    }

    private static double[] ComputeBlXPoints(int nPoints)
    {
        // Midpoints of an evenly spaced grid on [0, 1], matching compute_optimal_x_points.
        var points = new double[nPoints];
        for (var i = 0; i < nPoints; i++)
        {
            var left = (double)i / nPoints;
            var right = (double)(i + 1) / nPoints;
            points[i] = (left + right) / 2.0;
        }
        return points;
    }
}
