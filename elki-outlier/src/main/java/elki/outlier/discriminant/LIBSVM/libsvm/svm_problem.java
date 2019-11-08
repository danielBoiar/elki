package elki.outlier.discriminant.LIBSVM.libsvm;
public class svm_problem implements java.io.Serializable
{
	public int l;
	public double[] y;
	public svm_node[][] x;
	public int[] id;//for each point x[i] is id[i] the elki id. It is used for the score of a point
}
