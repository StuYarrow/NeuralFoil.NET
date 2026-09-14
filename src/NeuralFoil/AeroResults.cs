namespace NeuralFoil;

/// <summary>
/// Aerodynamic and boundary-layer results for one or more analysis cases. Every array is indexed by
/// case; boundary-layer arrays are further indexed by surface station (see <see cref="NeuralFoilModel.BlXPoints"/>).
/// </summary>
public sealed class AeroResults
{
    /// <summary>Number of analysis cases represented in this result.</summary>
    public int CaseCount { get; }

    /// <summary>Confidence of the network in its prediction; 1.0 is high confidence, 0.0 is low.</summary>
    public double[] AnalysisConfidence { get; }

    /// <summary>Lift coefficient.</summary>
    public double[] CL { get; }

    /// <summary>Drag coefficient.</summary>
    public double[] CD { get; }

    /// <summary>Moment coefficient.</summary>
    public double[] CM { get; }

    /// <summary>Transition location on the upper surface, as a fraction of chord (x/c).</summary>
    public double[] TopXtr { get; }

    /// <summary>Transition location on the lower surface, as a fraction of chord (x/c).</summary>
    public double[] BotXtr { get; }

    /// <summary>Upper-surface momentum thickness per station, indexed as [case][station].</summary>
    public double[][] UpperBlTheta { get; }

    /// <summary>Upper-surface boundary-layer shape factor per station, indexed as [case][station].</summary>
    public double[][] UpperBlH { get; }

    /// <summary>Upper-surface edge-velocity ratio (ue/vinf) per station, indexed as [case][station].</summary>
    public double[][] UpperBlUeOverVinf { get; }

    /// <summary>Lower-surface momentum thickness per station, indexed as [case][station].</summary>
    public double[][] LowerBlTheta { get; }

    /// <summary>Lower-surface boundary-layer shape factor per station, indexed as [case][station].</summary>
    public double[][] LowerBlH { get; }

    /// <summary>Lower-surface edge-velocity ratio (ue/vinf) per station, indexed as [case][station].</summary>
    public double[][] LowerBlUeOverVinf { get; }

    internal AeroResults(
        double[] analysisConfidence,
        double[] cl,
        double[] cd,
        double[] cm,
        double[] topXtr,
        double[] botXtr,
        double[][] upperBlTheta,
        double[][] upperBlH,
        double[][] upperBlUeOverVinf,
        double[][] lowerBlTheta,
        double[][] lowerBlH,
        double[][] lowerBlUeOverVinf)
    {
        this.CaseCount = analysisConfidence.Length;
        this.AnalysisConfidence = analysisConfidence;
        this.CL = cl;
        this.CD = cd;
        this.CM = cm;
        this.TopXtr = topXtr;
        this.BotXtr = botXtr;
        this.UpperBlTheta = upperBlTheta;
        this.UpperBlH = upperBlH;
        this.UpperBlUeOverVinf = upperBlUeOverVinf;
        this.LowerBlTheta = lowerBlTheta;
        this.LowerBlH = lowerBlH;
        this.LowerBlUeOverVinf = lowerBlUeOverVinf;
    }
}
