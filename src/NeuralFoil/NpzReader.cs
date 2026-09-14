using System.IO.Compression;
using System.Text;
using System.Text.RegularExpressions;

namespace NeuralFoil;

/// <summary>
/// A single N-dimensional array decoded from a NumPy <c>.npy</c> stream, flattened to
/// <see cref="double"/> values in C (row-major) order.
/// </summary>
public sealed class NumpyArray
{
    /// <summary>Create an array from its shape and flattened C-order data.</summary>
    public NumpyArray(int[] shape, double[] data)
    {
        this.Shape = shape;
        this.Data = data;
    }

    /// <summary>The array shape (e.g. <c>[128, 25]</c> for a weight matrix, <c>[128]</c> for a bias).</summary>
    public int[] Shape { get; }

    /// <summary>The flattened data in C (row-major) order.</summary>
    public double[] Data { get; }

    /// <summary>Number of dimensions.</summary>
    public int Rank => this.Shape.Length;

    /// <summary>Total number of elements.</summary>
    public int Count => this.Data.Length;

    /// <summary>Reinterpret a rank-2 array as a dense row-major matrix of shape (rows, cols).</summary>
    public double[,] AsMatrix()
    {
        if (this.Rank != 2)
            throw new InvalidOperationException($"Expected a rank-2 array, got rank {this.Rank}.");

        var rows = this.Shape[0];
        var cols = this.Shape[1];
        var result = new double[rows, cols];
        for (var r = 0; r < rows; r++)
            for (var c = 0; c < cols; c++)
                result[r, c] = this.Data[r * cols + c];
        return result;
    }
}

/// <summary>
/// Minimal reader for NumPy <c>.npy</c> arrays and <c>.npz</c> archives, sufficient to load the
/// NeuralFoil weight and distribution files. Supports little-endian <c>float32</c> (<c>&lt;f4</c>)
/// and <c>float64</c> (<c>&lt;f8</c>) data, in both C and Fortran order.
/// </summary>
public static class NpzReader
{
    private static readonly byte[] Magic = [0x93, (byte)'N', (byte)'U', (byte)'M', (byte)'P', (byte)'Y'];

    /// <summary>Load all named arrays from a <c>.npz</c> archive (a ZIP of <c>.npy</c> entries).</summary>
    public static Dictionary<string, NumpyArray> LoadNpz(string path)
    {
        using var stream = File.OpenRead(path);
        return LoadNpz(stream);
    }

    /// <summary>Load all named arrays from a <c>.npz</c> archive stream.</summary>
    private static Dictionary<string, NumpyArray> LoadNpz(Stream stream)
    {
        var result = new Dictionary<string, NumpyArray>(StringComparer.Ordinal);
        using var archive = new ZipArchive(stream, ZipArchiveMode.Read);
        foreach (var entry in archive.Entries)
        {
            if (!entry.FullName.EndsWith(".npy", StringComparison.OrdinalIgnoreCase))
                continue;

            var name = entry.FullName[..^".npy".Length];
            using var entryStream = entry.Open();
            // Zip entry streams are non-seekable; copy to memory so the parser can index freely.
            using var buffer = new MemoryStream();
            entryStream.CopyTo(buffer);
            buffer.Position = 0;
            result[name] = LoadNpy(buffer);
        }
        return result;
    }

    /// <summary>Decode a single NumPy <c>.npy</c> array from a stream.</summary>
    private static NumpyArray LoadNpy(Stream stream)
    {
        using var reader = new BinaryReader(stream, Encoding.ASCII, leaveOpen: true);

        var magic = reader.ReadBytes(Magic.Length);
        if (!magic.AsSpan().SequenceEqual(Magic))
            throw new InvalidDataException("Not a valid .npy file (bad magic string).");

        var versionMajor = reader.ReadByte();
        reader.ReadByte(); // version minor

        var headerLength = versionMajor >= 2
            ? (int)reader.ReadUInt32()
            : reader.ReadUInt16();

        var header = Encoding.ASCII.GetString(reader.ReadBytes(headerLength));
        var (description, fortranOrder, shape) = ParseHeader(header);

        var count = 1;
        foreach (var dim in shape)
            count *= dim;

        var data = ReadData(reader, description, count);

        if (fortranOrder && shape.Length > 1)
            data = FortranToC(data, shape);

        return new NumpyArray(shape, data);
    }

    private static (string descr, bool fortranOrder, int[] shape) ParseHeader(string header)
    {
        var descriptionMatch = Regex.Match(header, @"'descr'\s*:\s*'([^']+)'");
        var fortranMatch = Regex.Match(header, @"'fortran_order'\s*:\s*(True|False)");
        var shapeMatch = Regex.Match(header, @"'shape'\s*:\s*\(([^)]*)\)");

        if (!descriptionMatch.Success || !fortranMatch.Success || !shapeMatch.Success)
            throw new InvalidDataException($"Unrecognized .npy header: {header}");

        var description = descriptionMatch.Groups[1].Value;
        var fortranOrder = fortranMatch.Groups[1].Value == "True";

        var shape = shapeMatch.Groups[1].Value
            .Split(',')
            .Select(s => s.Trim())
            .Where(s => s.Length > 0)
            .Select(int.Parse)
            .ToArray();

        return (description, fortranOrder, shape);
    }

    private static double[] ReadData(BinaryReader reader, string description, int count)
    {
        var data = new double[count];
        switch (description)
        {
            case "<f4":
            case "|f4":
            {
                var bytes = reader.ReadBytes(count * sizeof(float));
                for (var i = 0; i < count; i++)
                    data[i] = ReadSingleLittleEndian(bytes, i * sizeof(float));
                break;
            }
            case "<f8":
            case "|f8":
            {
                var bytes = reader.ReadBytes(count * sizeof(double));
                for (var i = 0; i < count; i++)
                    data[i] = ReadDoubleLittleEndian(bytes, i * sizeof(double));
                break;
            }
            default:
                throw new NotSupportedException(
                    $"Unsupported .npy dtype '{description}'. Only little-endian float32 and float64 are supported.");
        }
        return data;
    }

    private static float ReadSingleLittleEndian(byte[] bytes, int offset)
    {
        if (BitConverter.IsLittleEndian)
            return BitConverter.ToSingle(bytes, offset);

        Span<byte> tmp = stackalloc byte[sizeof(float)];
        for (var i = 0; i < sizeof(float); i++)
            tmp[i] = bytes[offset + sizeof(float) - 1 - i];
        return BitConverter.ToSingle(tmp);
    }

    private static double ReadDoubleLittleEndian(byte[] bytes, int offset)
    {
        if (BitConverter.IsLittleEndian)
            return BitConverter.ToDouble(bytes, offset);

        Span<byte> tmp = stackalloc byte[sizeof(double)];
        for (var i = 0; i < sizeof(double); i++)
            tmp[i] = bytes[offset + sizeof(double) - 1 - i];
        return BitConverter.ToDouble(tmp);
    }

    private static double[] FortranToC(double[] fortran, int[] shape)
    {
        // Only rank-2 arrays appear in NeuralFoil's files; handle the general 2-D case.
        if (shape.Length != 2)
            throw new NotSupportedException("Fortran-order reshaping is only supported for rank-2 arrays.");

        var rows = shape[0];
        var cols = shape[1];
        var c = new double[fortran.Length];
        for (var r = 0; r < rows; r++)
            for (var col = 0; col < cols; col++)
                c[r * cols + col] = fortran[col * rows + r];
        return c;
    }
}
