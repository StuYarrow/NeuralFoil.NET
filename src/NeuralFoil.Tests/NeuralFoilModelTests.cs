namespace NeuralFoil.Tests;

using NUnit.Framework;

[TestFixture]
public class NeuralFoilModelTests
{
    private static KulfanParameters SampleAirfoil => new()
    {
        UpperWeights = [0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20],
        LowerWeights = [-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10],
        LeadingEdgeWeight = 0.15,
        TrailingEdgeThickness = 0.002,
    };

    [Test]
    public void AllModelSizes_LoadAndEvaluate()
    {
        foreach (var size in NeuralFoilModel.ModelSizes)
        {
            var aero = NeuralFoilModel.GetAeroFromKulfanParameters(
                SampleAirfoil, alpha: 3.0, re: 1e6, modelSize: size);

            Assert.That(aero.CaseCount, Is.EqualTo(1), size);
            Assert.That(double.IsFinite(aero.CL[0]), Is.True, $"{size} CL finite");
            Assert.That(double.IsFinite(aero.CD[0]), Is.True, $"{size} CD finite");
            Assert.That(aero.AnalysisConfidence[0], Is.InRange(0.0, 1.0), $"{size} confidence range");
        }
    }

    [Test]
    public void BatchedEvaluation_MatchesSingleCases()
    {
        double[] alphas = [-4.0, 0.0, 5.0, 10.0];
        var re = 5e5;

        var batched = NeuralFoilModel.GetAeroFromKulfanParameters(
            SampleAirfoil, alphas, [re], modelSize: "medium");

        Assert.That(batched.CaseCount, Is.EqualTo(alphas.Length));

        for (var i = 0; i < alphas.Length; i++)
        {
            var single = NeuralFoilModel.GetAeroFromKulfanParameters(
                SampleAirfoil, alpha: alphas[i], re: re, modelSize: "medium");

            Assert.That(batched.CL[i], Is.EqualTo(single.CL[0]).Within(1e-12), $"CL[{i}]");
            Assert.That(batched.CD[i], Is.EqualTo(single.CD[0]).Within(1e-12), $"CD[{i}]");
            Assert.That(batched.CM[i], Is.EqualTo(single.CM[0]).Within(1e-12), $"CM[{i}]");
            Assert.That(batched.AnalysisConfidence[i], Is.EqualTo(single.AnalysisConfidence[0]).Within(1e-12), $"conf[{i}]");
        }
    }

    [Test]
    public void BoundaryLayerOutputs_HaveExpectedShape()
    {
        var aero = NeuralFoilModel.GetAeroFromKulfanParameters(
            SampleAirfoil, alpha: 5.0, re: 1e6, modelSize: "medium");

        Assert.That(NeuralFoilModel.BlXPoints.Length, Is.EqualTo(NeuralFoilModel.N));
        Assert.That(aero.UpperBlTheta[0].Length, Is.EqualTo(NeuralFoilModel.N));
        Assert.That(aero.LowerBlUeOverVinf[0].Length, Is.EqualTo(NeuralFoilModel.N));
    }

    [Test]
    public void BlXPoints_AreMidpointsOfUniformGrid()
    {
        Assert.That(NeuralFoilModel.BlXPoints[0], Is.EqualTo(0.5 / NeuralFoilModel.N).Within(1e-12));
        Assert.That(NeuralFoilModel.BlXPoints[^1], Is.EqualTo(1.0 - 0.5 / NeuralFoilModel.N).Within(1e-12));
    }

    [Test]
    public void InvalidModelSize_Throws()
    {
        Assert.Throws<ArgumentException>(() =>
            NeuralFoilModel.GetAeroFromKulfanParameters(SampleAirfoil, 5.0, 1e6, modelSize: "gigantic"));
    }

    [Test]
    public void WrongWeightCount_Throws()
    {
        var bad = new KulfanParameters
        {
            UpperWeights = [0.1, 0.2, 0.3],
            LowerWeights = [-0.1, -0.2, -0.3],
        };
        Assert.Throws<ArgumentException>(() =>
            NeuralFoilModel.GetAeroFromKulfanParameters(bad, 5.0, 1e6));
    }

    [Test]
    public void MismatchedArrayLengths_Throw()
    {
        Assert.Throws<ArgumentException>(() =>
            NeuralFoilModel.GetAeroFromKulfanParameters(
                SampleAirfoil, [1.0, 2.0, 3.0], [1e6, 2e6], modelSize: "medium"));
    }
}
