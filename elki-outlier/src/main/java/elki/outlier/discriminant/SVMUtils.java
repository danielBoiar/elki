package elki.outlier.discriminant;
import elki.outlier.discriminant.LIBSVM.libsvm.*;
/**
 *
 * From LibSVM Kernel to calculate Kernel
 */

public class SVMUtils {
    static double dot(svm_node[] x, svm_node[] y)
    {
        double sum = 0;
        int xlen = x.length;
        int ylen = y.length;
        int i = 0;
        int j = 0;
        while(i < xlen && j < ylen)
        {
            if(x[i].index == y[j].index)
                sum += x[i++].value * y[j++].value;
            else
            {
                if(x[i].index > y[j].index)
                    ++j;
                else
                    ++i;
            }
        }
        return sum;
    }

    private static double powi(double base, int times)
    {
        double tmp = base, ret = 1.0;

        for(int t=times; t>0; t/=2)
        {
            if(t%2==1) ret*=tmp;
            tmp = tmp * tmp;
        }
        return ret;
    }

    static double k_function(svm_node[] x, svm_node[] y,
                             svm_parameter param)
    {
        switch(param.kernel_type)
        {
            case svm_parameter.LINEAR:
                return dot(x,y);
            case svm_parameter.POLY:
                return powi(param.gamma*dot(x,y)+param.coef0,param.degree);
            case svm_parameter.RBF:
            {
                double sum = 0;
                int xlen = x.length;
                int ylen = y.length;
                int i = 0;
                int j = 0;
                while(i < xlen && j < ylen)
                {
                    if(x[i].index == y[j].index)
                    {
                        double d = x[i++].value - y[j++].value;
                        sum += d*d;
                    }
                    else if(x[i].index > y[j].index)
                    {
                        sum += y[j].value * y[j].value;
                        ++j;
                    }
                    else
                    {
                        sum += x[i].value * x[i].value;
                        ++i;
                    }
                }

                while(i < xlen)
                {
                    sum += x[i].value * x[i].value;
                    ++i;
                }

                while(j < ylen)
                {
                    sum += y[j].value * y[j].value;
                    ++j;
                }

                return Math.exp(-param.gamma*sum);
            }
            case svm_parameter.SIGMOID:
                return Math.tanh(param.gamma*dot(x,y)+param.coef0);
            case svm_parameter.PRECOMPUTED:
                return	x[(int)(y[0].value)].value;
            default:
                return 0;	// java
        }
    }

    public static double calculateNorm_w(svm_model model) {
        //estimate norm w for rescaling score
        double[] sv_coef = model.sv_coef[0];
        double squared_norm_w = 0;
        for(int i=0;i<sv_coef.length;i++) {
            for(int j=0;j<sv_coef.length;j++) {
                squared_norm_w += sv_coef[i]*sv_coef[j] * SVMUtils.k_function(model.SV[i], model.SV[j], model.param);
            }
        }
        return Math.sqrt(squared_norm_w);
    }
}
