using NUnit.Framework;

namespace NeuralFoil.Tests;

[TestFixture]
public class GoldenValueTests
{
    // A fixed, mildly-cambered test airfoil defined directly in Kulfan parameters, matching
    // tests/test_golden_values.py so the C# port can be validated against the Python reference.
    private static KulfanParameters Kulfan => new()
    {
        UpperWeights = [0.20, 0.25, 0.20, 0.27, 0.20, 0.25, 0.20, 0.20],
        LowerWeights = [-0.10, -0.05, -0.10, 0.00, 0.05, 0.05, 0.10, 0.10],
        LeadingEdgeWeight = 0.15,
        TrailingEdgeThickness = 0.002,
    };

    private const double Rel = 1e-6;
    private const double Abs = 1e-9;

    private static readonly Dictionary<string, Dictionary<string, double>> GoldenKulfan = new()
    {
        ["medium"] = new()
        {
            ["analysis_confidence"] = 0.955711837783,
            ["CL"] = 1.10332809679,
            ["CD"] = 0.00919882438456,
            ["CM"] = -0.110598030451,
            ["Top_Xtr"] = 0.250543496781,
            ["Bot_Xtr"] = 0.964878409058,
        },
        ["xxxlarge"] = new()
        {
            ["analysis_confidence"] = 0.971352552774,
            ["CL"] = 1.09450207214,
            ["CD"] = 0.00911818948927,
            ["CM"] = -0.108654837628,
            ["Top_Xtr"] = 0.253019285687,
            ["Bot_Xtr"] = 1.0,
        },
    };

    [TestCase("medium")]
    [TestCase("xxxlarge")]
    public void GoldenKulfanDirect_MatchesPythonReference(string modelSize)
    {
        var aero = NeuralFoilModel.GetAeroFromKulfanParameters(
            Kulfan, alpha: 5.0, re: 1e6, modelSize: modelSize);

        var expected = GoldenKulfan[modelSize];
        AssertClose(expected["analysis_confidence"], aero.AnalysisConfidence[0], "analysis_confidence");
        AssertClose(expected["CL"], aero.CL[0], "CL");
        AssertClose(expected["CD"], aero.CD[0], "CD");
        AssertClose(expected["CM"], aero.CM[0], "CM");
        AssertClose(expected["Top_Xtr"], aero.TopXtr[0], "Top_Xtr");
        AssertClose(expected["Bot_Xtr"], aero.BotXtr[0], "Bot_Xtr");
    }

    private static void AssertClose(double expected, double actual, string label)
    {
        var tolerance = Abs + Rel * Math.Abs(expected);
        Assert.That(actual, Is.EqualTo(expected).Within(tolerance), label);
    }
}
