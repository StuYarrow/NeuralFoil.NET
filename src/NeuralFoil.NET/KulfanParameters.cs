namespace NeuralFoil;

/// <summary>
/// The Kulfan (CST) parameters describing an airfoil shape, using NeuralFoil's fixed resolution of
/// 8 CST weights per side plus a leading-edge modification weight and a trailing-edge thickness.
/// </summary>
public sealed class KulfanParameters
{
    /// <summary>The 8 CST weights for the upper surface, leading edge to trailing edge.</summary>
    public double[] UpperWeights { get; set; } = [];

    /// <summary>The 8 CST weights for the lower surface, leading edge to trailing edge.</summary>
    public double[] LowerWeights { get; set; } = [];

    /// <summary>The leading-edge modification (LEM) weight.</summary>
    public double LeadingEdgeWeight { get; set; }

    /// <summary>The trailing-edge thickness, in y/c units.</summary>
    public double TrailingEdgeThickness { get; set; }

    internal void Validate()
    {
        if (this.UpperWeights is null || this.UpperWeights.Length != 8)
            throw new ArgumentException(
                $"NeuralFoil expects exactly 8 CST weights per side, but `{nameof(this.UpperWeights)}` has " +
                $"{this.UpperWeights?.Length ?? 0}.");
        if (this.LowerWeights is null || this.LowerWeights.Length != 8)
            throw new ArgumentException(
                $"NeuralFoil expects exactly 8 CST weights per side, but `{nameof(this.LowerWeights)}` has " +
                $"{this.LowerWeights?.Length ?? 0}.");
    }
}
