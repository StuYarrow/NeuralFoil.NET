namespace NeuralFoil.Tests;

using NUnit.Framework;

using NeuralFoil;

[TestFixture]
public class NpzReaderTests
{
    [Test]
    public void LoadNpz_DistributionFile_HasExpectedArrays()
    {
        var dir = Utilities.ResolveDefaultWeightsDirectory();
        var arrays = NpzReader.LoadNpz(Path.Combine(dir, "scaled_input_distribution.npz"));

        Assert.That(arrays.ContainsKey("mean_inputs_scaled"), Is.True);
        Assert.That(arrays.ContainsKey("inv_cov_inputs_scaled"), Is.True);
        Assert.That(arrays["mean_inputs_scaled"].Count, Is.EqualTo(25));
        Assert.That(arrays["inv_cov_inputs_scaled"].Shape, Is.EqualTo(new[] { 25, 25 }));
    }

    [Test]
    public void LoadNpz_WeightFile_HasConsistentLayerShapes()
    {
        var dir = Utilities.ResolveDefaultWeightsDirectory();
        var arrays = NpzReader.LoadNpz(Path.Combine(dir, "nn-medium.npz"));

        var weight0 = arrays["net.0.weight"];
        var bias0 = arrays["net.0.bias"];

        Assert.That(weight0.Rank, Is.EqualTo(2));
        Assert.That(weight0.Shape[1], Is.EqualTo(25), "first layer takes the 25-dim input");
        Assert.That(bias0.Shape[0], Is.EqualTo(weight0.Shape[0]), "bias length equals layer width");
    }
}
