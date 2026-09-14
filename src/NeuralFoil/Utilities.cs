namespace NeuralFoil;

/// <summary>
/// Utility functions.
/// </summary>
public static class Utilities
{
    /// <summary>
    /// Locates the directory containing the NumPy archives of model weights.
    /// </summary>
    /// <returns>The path to the weights directory.</returns>
    /// <exception cref="DirectoryNotFoundException"></exception>
    public static string ResolveDefaultWeightsDirectory()
    {
        const string relativePath = "nn_weights_and_biases";

        // Walk up from the assembly location and the working directory looking for the distributed weights.
        var startPoints = new[]
        {
            AppContext.BaseDirectory,
            Path.GetDirectoryName(typeof(WeightRepository).Assembly.Location),
            Directory.GetCurrentDirectory(),
        };

        foreach (var start in startPoints)
        {
            if (string.IsNullOrEmpty(start))
                continue;

            for (var dir = new DirectoryInfo(start); dir is not null; dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, relativePath);
                if (Directory.Exists(candidate))
                    return candidate;
            }
        }

        throw new DirectoryNotFoundException(
            $"Could not locate the '{relativePath}' directory relative to the assembly or working directory. " +
            "Pass an explicit weights directory instead.");
    }
}