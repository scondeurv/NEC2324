using System.Text;
using Aglomera;

namespace AHC;

public class DataPoint : IEquatable<DataPoint>, IComparable<DataPoint>
{
    public DataPoint(string id, double[] value)
    {
        this.Id = id;
        this.Value = value;
    }

    public DataPoint()
    {
    }

    public string Id { get; }

    public double[] Value { get; }
    
    public override bool Equals(object obj) => obj is DataPoint && this.Equals((DataPoint)obj);

    public override int GetHashCode() => this.Id.GetHashCode();

    public override string ToString() => this.Id;

    public static bool operator ==(DataPoint left, DataPoint right) => left.Equals(right);

    public static bool operator !=(DataPoint left, DataPoint right) => !left.Equals(right);

    public int CompareTo(DataPoint other) => string.Compare(this.Id, other.Id, StringComparison.Ordinal);

    public bool Equals(DataPoint other) => string.Equals(this.Id, other.Id);
}