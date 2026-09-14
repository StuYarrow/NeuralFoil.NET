namespace NeuralFoil;

using System.Collections.Concurrent;

using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;


/// <summary>A single dense (fully-connected) layer: <c>y = W·x + b</c>.</summary>
internal sealed class DenseLayer(Matrix<double> weight, Vector<double> bias)
{
    public Matrix<double> Weight { get; } = weight;
    public Vector<double> Bias { get; } = bias;
}

/// <summary>The ordered dense layers of one NeuralFoil MLP.</summary>
internal sealed class NeuralNetwork(IReadOnlyList<DenseLayer> layers)
{
    public IReadOnlyList<DenseLayer> Layers { get; } = layers;
}

/// <summary>
/// Training-distribution statistics used to compute the Mahalanobis-based analysis confidence.
/// The inverse covariance is stored pre-inverted in the distribution file, so no runtime inversion occurs.
/// </summary>
internal sealed class InputDistribution(Vector<double> mean, Matrix<double> inverseCovariance)
{
    public Vector<double> Mean { get; } = mean;
    public Matrix<double> InverseCovariance { get; } = inverseCovariance;
    public int InputCount => this.Mean.Count;
}

/// <summary>
/// Loads and caches NeuralFoil weight files (<c>nn-{size}.npz</c>) and the input-distribution file
/// (<c>scaled_input_distribution.npz</c>). The weights directory is resolved automatically relative to the
/// loaded assembly (Option C) unless an explicit path is supplied.
/// </summary>
internal sealed class WeightRepository
{
    private static readonly Lazy<WeightRepository> DefaultInstance = new(() => new WeightRepository(Utilities.ResolveDefaultWeightsDirectory()));

    private readonly string weightsDirectory;

    private readonly ConcurrentDictionary<string, NeuralNetwork> networks = new(StringComparer.Ordinal);

    private InputDistribution? distribution;

    public WeightRepository(string weightsDirectory)
    {
        if (!Directory.Exists(weightsDirectory))
            throw new DirectoryNotFoundException($"NeuralFoil weights directory not found: {weightsDirectory}");
        this.weightsDirectory = weightsDirectory;
    }

    /// <summary>The shared repository that resolves the distributed weights directory automatically.</summary>
    public static WeightRepository Default => DefaultInstance.Value;

    public NeuralNetwork GetNetwork(string modelSize)
    {
        return this.networks.GetOrAdd(modelSize, size =>
        {
            var path = Path.Combine(this.weightsDirectory, $"nn-{size}.npz");
            if (!File.Exists(path))
                throw new ArgumentException(
                    $"Invalid model_size '{size}'. No weight file found at '{path}'.");
            return BuildNetwork(NpzReader.LoadNpz(path));
        });
    }

    public InputDistribution GetDistribution()
    {
        return this.distribution ??= BuildDistribution(
            NpzReader.LoadNpz(Path.Combine(this.weightsDirectory, "scaled_input_distribution.npz")));
    }

    private static NeuralNetwork BuildNetwork(Dictionary<string, NumpyArray> arrays)
    {
        // Keys look like "net.0.weight", "net.0.bias", "net.2.weight", ... (even indices are dense layers).
        var layerIndices = arrays.Keys
            .Where(k => k.StartsWith("net.", StringComparison.Ordinal))
            .Select(k => k.Split('.'))
            .Where(parts => parts.Length == 3 && int.TryParse(parts[1], out _))
            .Select(parts => int.Parse(parts[1]))
            .Distinct()
            .OrderBy(i => i)
            .ToList();

        if (layerIndices.Count == 0)
            throw new InvalidDataException(
                "Unexpected weight-file format: expected keys like 'net.0.weight', 'net.0.bias'.");

        var layers = new List<DenseLayer>(layerIndices.Count);
        foreach (var i in layerIndices)
        {
            var weight = arrays[$"net.{i}.weight"];
            var bias = arrays[$"net.{i}.bias"];
            layers.Add(new DenseLayer(
                DenseMatrix.OfArray(weight.AsMatrix()),
                DenseVector.OfArray(bias.Data)));
        }

        return new NeuralNetwork(layers);
    }

    private static InputDistribution BuildDistribution(Dictionary<string, NumpyArray> arrays)
    {
        return new InputDistribution(
            DenseVector.OfArray(arrays["mean_inputs_scaled"].Data),
            DenseMatrix.OfArray(arrays["inv_cov_inputs_scaled"].AsMatrix()));
    }
}
