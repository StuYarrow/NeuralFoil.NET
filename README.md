# NeuralFoil.NET
A .NET class library implementing the core inference functionality of [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil).

### Changelog

#### v1.0.6
- Make weights files non-visible in consuming projects.

#### v1.0.5
- Move weights files into separate directory.

#### v1.0.4
- Initial release.

### Usage

1. Create a `KulfanParameters` object containing the aerofoil profile in Kulfan/CST form.
For more information on fitting Kulfan parameters to a polyline aerofoil profile, see [AeroSandbox](https://github.com/peterdsharpe/AeroSandbox).
2. Call one of the `NeuralFoilModel.GetAeroFromKulfanParameters()` entry points, passing the Kulfan aerofoil definition and aero case parameters.
An `AeroResults` object containing the results is returned.

See [NeuralFoil](https://github.com/peterdsharpe/NeuralFoil) docs for further information.

### Contributors

NeuralFoil.NET is a direct port of [NeuralFoil's](https://github.com/peterdsharpe/NeuralFoil) inference kernel to C#, and repackages NeuralFoil's original NumPy archives containing the model weights.
NeuralFoil is written and trained by [Peter Sharpe](https://peterdsharpe.github.io/).

The port is maintained by Stu Yarrow with LLM support.
Work on this port is supported by [Ozone](https://flyozone.com/).