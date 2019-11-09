/*
 * This file is part of ELKI:
 * Environment for Developing KDD-Applications Supported by Index-Structures
 *
 * Copyright (C) 2019
 * ELKI Development Team
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
 * GNU Affero General Public License for more details.
 *
 * You should have received a copy of the GNU Affero General Public License
 * along with this program. If not, see <http://www.gnu.org/licenses/>.
 */
package elki.outlier.discriminant;

import elki.outlier.OutlierAlgorithm;
import elki.outlier.discriminant.LIBSVM.libsvm.*;
import elki.data.NumberVector;
import elki.data.type.TypeInformation;
import elki.data.type.TypeUtil;
import elki.database.datastore.DataStoreFactory;
import elki.database.datastore.DataStoreUtil;
import elki.database.datastore.WritableDoubleDataStore;
import elki.database.ids.*;
import elki.database.relation.DoubleRelation;
import elki.database.relation.MaterializedDoubleRelation;
import elki.database.relation.Relation;
import elki.database.relation.RelationUtil;
import elki.logging.Logging;
import elki.math.DoubleMinMax;
import elki.result.outlier.BasicOutlierScoreMeta;
import elki.result.outlier.OutlierResult;
import elki.result.outlier.OutlierScoreMeta;
import elki.utilities.exceptions.AbortException;
import elki.utilities.optionhandling.OptionID;
import elki.utilities.optionhandling.Parameterizer;
import elki.utilities.optionhandling.constraints.GreaterConstraint;
import elki.utilities.optionhandling.constraints.GreaterEqualConstraint;
import elki.utilities.optionhandling.constraints.LessConstraint;
import elki.utilities.optionhandling.constraints.LessEqualConstraint;
import elki.utilities.optionhandling.parameterization.Parameterization;
import elki.utilities.optionhandling.parameters.*;
import java.util.*;



/**
 * Leave-One-Out Outlier-detection using support vector machines One Class SVM and Support Vector Data Description.
 * <p>
 * Leaves one point out of the trainingsset and estimates the distance between the leftout point and the boundary to calculate the score. 
 * This is repeated for all points. 
 * <p>
 * 
 * @author Daniel Boiar
 * @since 0.7.6
 *
 * @param <V> vector type
 */
public class LeaveMoreOutLibSVMOneClassOutlierDetection<V extends NumberVector> implements OutlierAlgorithm {
  /**
   * Class logger.
   */
  private static final Logging LOG = Logging.getLogger(LeaveMoreOutLibSVMOneClassOutlierDetection.class);
  
  /**
   * SVM types. Expose as enum for convenience.
   */
  public enum SVMType { //
    ONE_CLASS, // one class svm
    SVDD, // support vector data description
  }
  
  /**
   * Kernel functions. Expose as enum for convenience.
   */
  public enum SVMKernel { //
    LINEAR, // Linear
    QUADRATIC, // Quadratic
    CUBIC, // Cubic
    RBF, // Radial basis functions
    SIGMOID, // Sigmoid
  }

  /**
   * Kernel function in use.
   */
  protected SVMType type = SVMType.ONE_CLASS;
  
  /**
   * Kernel function in use.
   */
  protected SVMKernel kernel = SVMKernel.RBF;

  /**
   * Nu parameter for OCSVM.
   */
  protected double nu = 0.05;
  
  /**
   * C parameter for SVDD.
   */
  protected double C = 1.0;
  
  /**
   * Gamma parameter.
   */
  protected double gamma = 0.5;//1.0/dim
  
  /**
   * Pathdepth parameter.
   */
  protected int pathdepth = 1;

  /**
   * AlphaDistributionMethod parameter.
   */
  protected alphaDistributionMethod alphaDistributionMethodParameter = alphaDistributionMethod.RANDOM;

  /**
   * AggregateMethod parameter.
   */
  protected aggregateMethod aggregateMethodParameter = aggregateMethod.WEIGHTED_AVERAGE;

  /**
   * NextLeaveMoreOutElementMethod parameter.
   */
  protected nextLeaveMoreOutElementMethod nextLeaveMoreOutElementMethodParameter = nextLeaveMoreOutElementMethod.RANDOM;

  
  /**
   * Zeta parameter.
   */
  protected double zeta = 0.7;
  
  /**
   * AlphaDistributionMethod seed.
   */
  protected long alphaDistributionMethodSeed = Long.parseLong("4321"); 
  
  /**
   * NextLeaveMoreOutElementMethod seed.
   */
  protected long nextLeaveMoreOutElementMethodSeed = Long.parseLong("5432"); 

  /**
   * Activate random sampling.
   */
  protected boolean randomSampling = false;
  
  /**
   * RandomSampling seed.
   */
  protected long randomSamplingSeed = Long.parseLong("4567"); 
  
  /**
   * Activate the alternating strategy that alternates between random sampling and nextLeaveMoreOutStrategy.
   */
  protected boolean alternateRandomSampling = true;
  
  /**
   * RandomSamplingDepth parameter.
   */
  protected int countRandomSamplingDepth = 5;
  
  /**
   * Constructor.
   *
   * @param kernel Kernel to use with SVM.
   * @param nu Nu parameter
   * @param gamma Gamma parameter
   * @param pathdepth Pathdepth parameter
   */
  public LeaveMoreOutLibSVMOneClassOutlierDetection(SVMType type, SVMKernel kernel, double nu, double C, double gamma, int pathDepth, alphaDistributionMethod alphaDistributionMethodParameter, aggregateMethod aggregateMethodParameter, nextLeaveMoreOutElementMethod nextLeaveMoreOutElementMethodParameter, double zeta, long alphaDistributionMethodSeed, long nextLeaveMoreOutElementMethodSeed, boolean randomSampling, long randomSamplingSeed, boolean alternateRandomSampling, int countRandomSamplingDepth) {
    super();
    this.type = type;
    this.kernel = kernel;
    this.nu = nu;
    this.C = C;
    this.gamma = gamma;
    this.pathdepth = pathDepth;
    this.alphaDistributionMethodParameter = alphaDistributionMethodParameter;
    this.aggregateMethodParameter = aggregateMethodParameter;
    this.nextLeaveMoreOutElementMethodParameter = nextLeaveMoreOutElementMethodParameter;
    this.zeta = zeta;
    this.alphaDistributionMethodSeed = alphaDistributionMethodSeed;
    this.nextLeaveMoreOutElementMethodSeed = nextLeaveMoreOutElementMethodSeed;
    this.randomSampling = randomSampling; 
    this.randomSamplingSeed = randomSamplingSeed; 
    this.alternateRandomSampling = alternateRandomSampling; 
    this.countRandomSamplingDepth = countRandomSamplingDepth;
  }

  /**
   * Run LeaveOneOut one-class SVM.
   * 
   * @param relation Data relation
   * @return Outlier result.
   */
  public OutlierResult run(Relation<V> relation) {
    final int dim = RelationUtil.dimensionality(relation);
    final ArrayDBIDs ids = DBIDUtil.ensureArray(relation.getDBIDs()); 
    
    svm.svm_set_print_string_function(LOG_HELPER);

    svm_parameter param = new svm_parameter();
    param.svm_type= svm_parameter.ONE_CLASS;
    switch(type){
    case ONE_CLASS:
      param.svm_type = svm_parameter.ONE_CLASS;
      break;
    case SVDD:
      param.svm_type = svm_parameter.SVDD;
      break;
    default:
      throw new AbortException("Invalid svm type parameter: " + type);
    }
    
    param.kernel_type = svm_parameter.LINEAR;
    param.degree = 3;
    switch(kernel){
    case LINEAR:
      param.kernel_type = svm_parameter.LINEAR;
      break;
    case QUADRATIC:
      param.kernel_type = svm_parameter.POLY;
      param.degree = 2;
      break;
    case CUBIC:
      param.kernel_type = svm_parameter.POLY;
      param.degree = 3;
      break;
    case RBF:
      param.kernel_type = svm_parameter.RBF;
      break;
    case SIGMOID:
      param.kernel_type = svm_parameter.SIGMOID;
      break;
    default:
      throw new AbortException("Invalid kernel parameter: " + kernel);
    }
    param.nu = nu;
    param.coef0 = 0.;
    param.cache_size = 10000;
    param.C = C; //used by one-class with C=1 because of scaled version in libsvm, only for svdd! use nu for ocsvm instead. UpperBound is indepent from nu, because Cpositiv = 1 at runSolver.
    param.eps = 1e-4;//1e-4; // stopping criteria used by one-class or svdd in Solver
    param.p = 0.1; // not used by one-class or svdd! -> p is only used in epsilon svr.
    param.shrinking = 0;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = new int[0];
    param.weight = new double[0];
    param.gamma = gamma; // 1.0/dim;

    // Transform data:
    svm_problem prob = new svm_problem();
    prob.l = relation.size();
    prob.x = new svm_node[prob.l][];
    prob.y = new double[prob.l];
    prob.id = new int[prob.l];
    {
      DBIDIter iter = ids.iter();
      for(int i = 0; i < prob.l && iter.valid(); iter.advance(), i++) {
        V vec = relation.get(iter);
        svm_node[] x = new svm_node[dim];
        for(int d = 0; d < dim; d++) {
          x[d] = new svm_node();
          x[d].index = d + 1;
          x[d].value = vec.doubleValue(d);
        }
        prob.x[i] = x;
        prob.y[i] = +1;
        prob.id[i] = ids.binarySearch(iter); 
      }
    }
    if(LOG.isVerbose()) {
      LOG.verbose("Training "+ type.toString() +" SVM...");
    }
    
    String err = svm.svm_check_parameter(prob, param);
    if(err != null) {
      LOG.warning("svm_check_parameter: " + err);
    }
    long startTime;
    long endTime;
    startTime = System.currentTimeMillis();
    svm svmManager = new svm(prob, param);
    svm_model model = svmManager.svm_trainGivenCache(prob,param); //creates cache and solves OCSVM for all points. Therefore Kernels are estimated
    endTime = System.currentTimeMillis();
    if (LOG.isVerbose()) {LOG.verbose("Time at learning all: " + "That took " + (endTime - startTime)/1000 + " seconds");}
    
    //Leave One Out: Train without the t-th Sample
    WritableDoubleDataStore scores = DataStoreUtil.makeDoubleStorage(relation.getDBIDs(), DataStoreFactory.HINT_DB);
    DoubleMinMax mm = new DoubleMinMax();
    DBIDIter iter = ids.iter();

    double[] buf = new double[svm.svm_get_nr_class(model)];

    int smallestPossibleIndex = 0;
    int[] svIndices = model.sv_indices;
    Arrays.sort(svIndices);
    boolean init = true; // for first init Training
    svm_model model_t = null;
    svm_problem prob_t = new svm_problem();
    int old_t = -1;
    svm_node[] old_x = null;
    int old_id = -1;
    NextOldLeftOutElement nextOldLeftOutElement;
    int pathDepth = this.pathdepth; //h
    
    //backup G
    double[] G = new double[model.Gradient.length]; 
    double[] G_bar = new double[model.Gradient_bar.length];
    double[] G_t = new double[model.Gradient.length]; 
    double[] G_bar_t = new double[model.Gradient_bar.length];
    // clone G and G_bar 
    for (int i = 0; i < model.Gradient.length; i++) {
      G[i] = model.Gradient[i];
      G_bar[i] = model.Gradient_bar[i];
    }
    for(int t = 0; t < prob.l && iter.valid(); iter.advance(), t++) {
      if (LOG.isVerbose()){LOG.verbose("Training one-class SVM with " + t + "-th Sample removed...");}
      startTime = System.currentTimeMillis();
      //getNextSVIndex
      int nextSVIndex;
      if (smallestPossibleIndex < svIndices.length) {
        nextSVIndex = svIndices[smallestPossibleIndex]; // only SV could be outliers -> check only the SV
      } else {
        nextSVIndex = -1;
      }

      //Init: only at first training for initialization
      if (init) {        
        nextOldLeftOutElement = createProb_t(prob, prob_t, t);
        old_x = nextOldLeftOutElement.getNew_x();
        old_id = nextOldLeftOutElement.getNew_id();
        old_t = t;
        model_t = svmManager.svm_trainGivenCache(prob_t, param);//securing init correctness
        model_t.param.svm_type = param.svm_type;
        init = false;
        if (t == nextSVIndex - 1) {//Only if t is SV
          smallestPossibleIndex++; //If 1 in SV_indices, then nextSVindex needs to be updated in init case
        }
      }else { // no init
        if (t == nextSVIndex - 1) {//Only if t is SV
          if (LOG.isVerbose()) {LOG.verbose("t is SV at " + t + "!");}
          
          //rollout G
          G_t = G.clone();
          G_bar_t = G_bar.clone();
          
          
          //copy alpha with size--
          double[] alpha = new double[model.sv_indices.length - 1];
          int[] svIndices_t = new int[model.sv_indices.length - 1];
          double alpha_t = model.sv_coef[0][smallestPossibleIndex];
          if(alpha_t <= 0) {
            if (LOG.isVerbose()) {LOG.verbose("Error at sample " + t + "! alpha_t <= 0");}
            System.exit(-2);
          }
          if(prob.l - 1 == model.sv_indices[alpha.length] - 1){//test if last position is SV
            //last position is SV -> swap -> only delete last position because: last position becomes t
            for(int i = 0; i< alpha.length ;i++) { // only till alpha.length - 1
              alpha[i] = model.sv_coef[0][i];
              svIndices_t[i] = model.sv_indices[i];
            }
            if(smallestPossibleIndex < alpha.length) {
              alpha[smallestPossibleIndex] = model.sv_coef[0][alpha.length]; //swap the alpha values, but in the sv_indices there is not swap
            }
          }else {//last position is not a SV -> only delete smallest Possible Index
            for (int i = 0; i < model.sv_coef[0].length; i++) {
              if (i < smallestPossibleIndex) {
                alpha[i] = model.sv_coef[0][i];
                svIndices_t[i] = model.sv_indices[i];
              } else if (i > smallestPossibleIndex) {
                alpha[i - 1] = model.sv_coef[0][i];
                svIndices_t[i - 1] = model.sv_indices[i];//[i]-1 ?
              }
            }
          }

          updateAlphaAndGradient(param, prob, svmManager, G_t,G_bar_t,prob.l, smallestPossibleIndex, model_t, alphaDistributionMethodParameter, t, alpha, svIndices_t, alpha_t);
          nextOldLeftOutElement = createProb_t_WithoutCopy(prob_t, t, old_t, old_x, old_id);
          if(nextOldLeftOutElement != null) { //only null if t is last position
            old_x = nextOldLeftOutElement.getNew_x();
            old_id = nextOldLeftOutElement.getNew_id();
            old_t = t;
          }
          svmManager.swapCache(t, prob.l - 1);
          model_t = svmManager.svm_trainGivenAlphaAndCache(prob_t,param,model_t); 
          svmManager.swapCache(t, prob.l - 1);
          if (LOG.isVerbose()) {
            String headerRound = "round,";
            String deletedPoint = "0,";
            for(int dimension = 0; dimension < dim - 1; dimension++){
              headerRound += "x"+dimension+",";
              deletedPoint += prob.x[t][dimension].value+",";
            }
            headerRound += "x"+(dim - 1);
            deletedPoint += prob.x[t][dim - 1].value;
            LOG.verbose(headerRound);
            LOG.verbose(deletedPoint);
          }
          smallestPossibleIndex++;
        }
      }//end else init
      V vec = relation.get(iter);
      svm_node[] x = new svm_node[dim];
      for (int d = 0; d < dim; d++) {
        x[d] = new svm_node();
        x[d].index = d + 1;
        x[d].value = vec.doubleValue(d);
      }
      double score = getScore(x, dim, model, buf, model_t, t, nextSVIndex);
      if (t == nextSVIndex - 1 && init == false) {//Only if t is SV
        if (LOG.isVerbose()) {LOG.verbose("Pathtracing in the " + t + "-th Sample...");}
        // Path tracing
        svm_model modelPath = cloneModel(model_t);//modelPath is the model with is updated in the path process
        modelPath.param = new svm_parameter();
        modelPath.param.svm_type = param.svm_type; // used in prediction
        svm_problem probOld = new svm_problem();
        cloneProb(prob_t,probOld);
        double[] pathScores = new double[pathDepth]; 
        pathScores[0] = score;
        int[] xIndexOnPath = new int[pathDepth]; // array that save the index of the removed element in the proportionate round. xIndexOnPath[round]
        xIndexOnPath[0] = t;
        svmManager.swapCache(t, prob.l - 1);
        Stack<int[]> swaps = new Stack<int[]>();// stack of tuples: save changed points
        int[] tupel = new int[2];
        tupel[0] = t;
        tupel[1] = prob.l - 1;
        swaps.push(tupel);
        
        int indexJ = 0;
        double alphaJ = 0;
        int round;

        /**
         *
         * Path
         *
         */
        boolean inRandomSamplingMode = true;// local parameter that is used if alternateRandomSampling is set to distinguish between Random Sampling mode and nextLeaveMoreOutElementMethod
        int countRandomSampling; // local loop parameter
        int userPathDepth = pathDepth; // local parameter for save the user parameter for a rollout, when pathDepth is overridden
        Queue<Integer> svAtBound = null; // contain SV at upper bound if random Sampling is set
        if(!randomSampling) {
          countRandomSamplingDepth = 1; // deactivates random sampling 
        }
        for (countRandomSampling = 0; countRandomSampling < countRandomSamplingDepth; countRandomSampling++) {
          if(randomSampling && inRandomSamplingMode) {
            //estimate SV to remove via random sampling
            double maxSlack = 0;
            Queue<Double> slacks = new LinkedList<>();
            //get indices of SV at upper bound
            svAtBound = new LinkedList<Integer>();
            for ( int i = 0; i < modelPath.sv_coef[0].length; i++) {
              if (modelPath.sv_coef[0][i] >= param.C) {
                //estimate slack and only add i with probability (1-slack)
                //predict score for estimate slack
                double[] bufSlack = new double[svm.svm_get_nr_class(model)];
                svm.svm_predict_values(modelPath, probOld.x[modelPath.sv_indices[i]-1], bufSlack);
                double slack = -bufSlack[0];
                if(slack < 0) {
                  if (LOG.isVerbose()) {LOG.verbose("WARNING! negative Slack");}                     
                }
                if(slack > maxSlack) {
                  maxSlack = slack;
                }
                slacks.add(slack); // FIFO guarantees order
              }
            }
            for ( int i = 0; i < modelPath.sv_coef[0].length; i++) {
              if (modelPath.sv_coef[0][i] >= param.C) {
                // only add i with probability (1-slack)
                double slack = slacks.poll();
                if(slack<0) {
                  slack = 0;
                }
                double slackProbability;
                if(maxSlack <= 0) {
                  slackProbability = 0;
                }else {
                  slackProbability = slack / maxSlack;
                }
                
                Random random = new Random(randomSamplingSeed);
                if( 1-slackProbability <= random.nextDouble() ) {
                  svAtBound.add(modelPath.sv_indices[i]);
                }
              }
            }
            if(inRandomSamplingMode) {
              pathDepth = svAtBound.size() + 1; //+1 because of start at round = 1
              xIndexOnPath = new int[pathDepth]; 
              xIndexOnPath[0] = t;
            }
          }//endif randomSampling
          
          for (round = 1; round < pathDepth; round++) {
              if(modelPath.sv_indices.length <= 1){//if pathDepth is too long
                if (LOG.isVerbose()) {LOG.verbose("WARNING! no SV in new model");}
                break;
              }
              if(randomSampling && inRandomSamplingMode) {
                int svIndicesJ = svAtBound.poll();
                //indexJ is the position in svIndices. Search for it.
                indexJ = -1;
                for(int i = 0; i < modelPath.sv_indices.length; i++) {
                  if( modelPath.sv_indices[i] == svIndicesJ ) {
                    indexJ = i;
                    break;
                  }
                }
                if(indexJ<0) {
                  if (LOG.isVerbose()) {LOG.verbose("WARNING! index negative");}
                  continue;
                }
                if (LOG.isVerbose()) {LOG.verbose("RandomSampling");}
              }else {
                indexJ = getIndexJ(relation, dim, ids, svmManager, scores, nextLeaveMoreOutElementMethodParameter, modelPath, probOld, swaps, round);
              }                
              xIndexOnPath[round] = modelPath.sv_indices[indexJ] - 1; //Element that will be removed
              
              svm_node[] xJ = probOld.x[xIndexOnPath[round]];
              alphaJ = modelPath.sv_coef[0][indexJ];  
  
              //distribute Alpha and update modelPath's Gradient
              //copy alpha with size--
              double[] alpha = new double[modelPath.sv_indices.length - 1]; 
              int[] svIndices_t = new int[modelPath.sv_indices.length - 1];
              
              if(probOld.l - 1 == modelPath.sv_indices[alpha.length] - 1){//test if last position is SV
                //last position is SV -> swap -> only delete last position because: last position becomes t
                for(int i = 0; i < alpha.length ;i++) { // only till alpha.length - 1
                  alpha[i] = modelPath.sv_coef[0][i];
                  svIndices_t[i] = modelPath.sv_indices[i];
                }
                if(indexJ < alpha.length) {
                  alpha[indexJ] = modelPath.sv_coef[0][alpha.length]; //swap the alpha values, but in the sv_indices there is not swap
                }
              }else {//last position is not a SV -> only delete smallest Possible Index
                for (int i = 0; i < modelPath.sv_coef[0].length; i++) {
                  if (i < indexJ) {
                    alpha[i] = modelPath.sv_coef[0][i];
                    svIndices_t[i] = modelPath.sv_indices[i];
                  } else if (i > indexJ) {
                    alpha[i - 1] = modelPath.sv_coef[0][i];
                    svIndices_t[i - 1] = modelPath.sv_indices[i];
                  }
                }
              }
  
              updateAlphaAndGradient(param, probOld,svmManager,modelPath.Gradient,modelPath.Gradient_bar,probOld.l,indexJ,modelPath,alphaDistributionMethodParameter,xIndexOnPath[round],alpha,svIndices_t,alphaJ);
              svmManager.swapCache(xIndexOnPath[round], probOld.l - 1);
              tupel = new int[2];
              tupel[0] = xIndexOnPath[round];
              tupel[1] = probOld.l-1;
              swaps.push(tupel);
              //delete xIndexOnPath[j] in probPath
              probOld.x[xIndexOnPath[round]] = probOld.x[probOld.l-1].clone(); //clone is needed!?
              int idJ = probOld.id[xIndexOnPath[round]];
              probOld.id[xIndexOnPath[round]] = probOld.id[probOld.l-1]; // id is swapped too 
              probOld.l--;
              if(!randomSampling || (randomSampling && !inRandomSamplingMode)) {
                modelPath = svmManager.svm_trainGivenAlphaAndCache(probOld, param, modelPath);
              }else {
                modelPath.l--;
              }
  
              /**
               *
               * Score and Aggregation
               *
               */
              if (LOG.isVerbose()) {
                String headerRound = "round,";
                String deletedPoint = round+",";
                for(int dimension = 0; dimension < dim - 1; dimension++){
                  headerRound += "x"+dimension+",";
                  deletedPoint += xJ[dimension].value+",";
                }
                headerRound += "x"+(dim - 1);
                deletedPoint += xJ[dim - 1].value;
                LOG.verbose(headerRound);
                LOG.verbose(deletedPoint);
              }
  
              //predict score
              double[] bufNew = new double[svm.svm_get_nr_class(model)];
              svm.svm_predict_values(modelPath, xJ, bufNew);
              pathScores[0] = -bufNew[0];// / normNew_w; //[rounds]wurde geaendert zu [0]
              
              DBIDRef iterJ = ids.get(idJ);
              double aggregatedScore = 0;
              switch (aggregateMethodParameter){//Aggregate Score Method is important for the leftoutmethod MaxScore
                case OVERRIDE://could be included into case Weighted_Average mit zeta = 1.0
                  aggregatedScore = pathScores[0];//pass through
                  break;
                case WEIGHTED_AVERAGE://exponential moving average ... see wikipedia
                  double oldScore = scores.doubleValue(iterJ); // was ist mit dem score, bei dem noch nichts initalisiert ist? -> zeta auf 0 setzen. Boolean Flag für jedes sample? -> es ist doch schon initialisiert durch den ersten SVM durchgang
                  //double zeta = 0.7; // user parameter
                  aggregatedScore = zeta * pathScores[0] + (1-zeta) * oldScore;
                  break;
              }
              scores.putDouble(iterJ,aggregatedScore);
              mm.put(aggregatedScore);
          }//ENDFOR rounds
          
          if(randomSampling && alternateRandomSampling) {
            if(!inRandomSamplingMode) { // a normal run without random sampling doesn't count as random sampling
              countRandomSampling--;
            }else {
              pathDepth = userPathDepth;
              xIndexOnPath = new int[pathDepth];
              xIndexOnPath[0] = t;
            }
            inRandomSamplingMode = !inRandomSamplingMode;
            
          }
        }
        while (!swaps.isEmpty()) {//reverse swap cache for all changes
          tupel = swaps.pop();
          svmManager.swapCache(tupel[0],tupel[1]);
        }


        if (LOG.isVerbose()) {LOG.verbose("End Pathtracing in the " + t + "-th Sample...");}
      }
       
      //for normal case with pathDepth == 1
      scores.putDouble(iter, score);// puts the score of t and not the pathPoints
      mm.put(score);
      endTime = System.currentTimeMillis();
      if (LOG.isVerbose()) {LOG.verbose("Time at sample "+t+": " + "That took " + (endTime - startTime)/1000 + " seconds");}
    }//end for t
    
    
    DoubleRelation scoreResult = new MaterializedDoubleRelation(type.toString()+" SVM Decision", ids, scores);
    OutlierScoreMeta scoreMeta = new BasicOutlierScoreMeta(mm.getMin(), mm.getMax(), -1.0, 1.0, 0.);
    return new OutlierResult(scoreMeta, scoreResult);
  }

  

  private void updateAlphaAndGradient(svm_parameter param, svm_problem prob, svm svmManager, double[] G, double[] G_bar, int l, int smallestPossibleIndex, svm_model model_t, alphaDistributionMethod alphaDistributionMethodParameter, int t, double[] alpha, int[] svIndices_t, double alpha_t) {
    float[] Q_i;
    double delta_alpha_i;
    switch (alphaDistributionMethodParameter){
      case RANDOM_NO_SV:
        Random random_NO_SV = new Random(alphaDistributionMethodSeed);
        int randomIndex_NO_SV = random_NO_SV.nextInt(prob.l - 1); // new strategy could be to pick only SV not at bound C
        int counter = 0;
        boolean allPointsAreSV = false;
        while( (randomIndex_NO_SV == smallestPossibleIndex && alpha.length > 1) || contains(svIndices_t,randomIndex_NO_SV) || randomIndex_NO_SV < 1 || randomIndex_NO_SV == t){ // new assignment if randomIndex is the leftout index
          counter++; // counts iterations
          randomIndex_NO_SV = random_NO_SV.nextInt(prob.l - 1); // try again
          if(counter>1000) {// break infinity loop (only if no value found for wakened)
            //probably all points are SV and no one can be woken up
            allPointsAreSV = true;
            break;
          }
        }
        if(!allPointsAreSV) {
          //Copy and resize
          int[] svIndices_t_wakened = new int[svIndices_t.length + 1]; 
          double[] alpha_t_wakened = new double[svIndices_t.length + 1];
          for(int i = 0; i < svIndices_t.length; i++) {
            svIndices_t_wakened[i] = svIndices_t[i];
            alpha_t_wakened[i] = alpha[i];
          }
          //give the random point all alpha of the left out point
          double oldAlphaRandomIndex_NO_SV = 0;
          alpha_t_wakened[svIndices_t.length] = alpha_t;
          svIndices_t_wakened[svIndices_t.length] = randomIndex_NO_SV;
    
          //update G
          delta_alpha_i = alpha_t - oldAlphaRandomIndex_NO_SV;
          double delta_alpha_j_NO_SV = 0 - alpha_t;
          Q_i = svmManager.getQ(randomIndex_NO_SV,G.length);
          float[] Q_j_NO_SV = svmManager.getQ(t,G.length);
          for(int k=0;k<G.length;k++)
          {
            G[k] += Q_i[k]*delta_alpha_i + Q_j_NO_SV[k]*delta_alpha_j_NO_SV;
          }
          // update G_bar
          if( (oldAlphaRandomIndex_NO_SV >= param.C && alpha_t_wakened[svIndices_t.length] < param.C) || (oldAlphaRandomIndex_NO_SV < param.C && alpha_t_wakened[svIndices_t.length] >= param.C)){ //upper bound changed check: see svm solver update alpha_status and G_bar
            if( oldAlphaRandomIndex_NO_SV >= param.C){ //is upper bound
                for(int j=0;j<G_bar.length;j++){
                    G_bar[j] -= param.C * Q_i[j]; //upper bound * Q_i[j]
                }
            }else{
                for(int j=0;j<G_bar.length;j++){
                    G_bar[j] += param.C * Q_i[j];//G_bar[model.sv_indices[i]-1] += Q_i[j];
                }
            }
          }
          if( alpha_t >= param.C){ //is upper bound
            for(int k=0;k<l;k++){
                G_bar[k] -= param.C * Q_j_NO_SV[k]; //upper bound * Q_i[j]
            }
          }else{
            for(int k=0;k<l;k++){
                G_bar[k] += param.C * Q_j_NO_SV[k]; 
            }
          }
          model_t.sv_indices = svIndices_t_wakened;
          model_t.sv_coef[0] = alpha_t_wakened;
          
          //swap max sv_indices to last position
          int maxIndices=-1;
          int indexOfMaxIndices = -1;
          for(int i = 0; i < model_t.sv_indices.length; i++) {
            if ( model_t.sv_indices[i] > maxIndices ) {
              maxIndices = model_t.sv_indices[i];
              indexOfMaxIndices = i;
            }  
          }
          model_t.sv_indices[indexOfMaxIndices] = model_t.sv_indices[model_t.sv_indices.length - 1];
          model_t.sv_indices[model_t.sv_indices.length - 1] = maxIndices;
          double tempAlpha = model_t.sv_coef[0][indexOfMaxIndices];
          model_t.sv_coef[0][indexOfMaxIndices] = model_t.sv_coef[0][model_t.sv_indices.length - 1];
          model_t.sv_coef[0][model_t.sv_indices.length - 1] = tempAlpha;
          
        } else { //all points are SV
          model_t.sv_coef[0] = alpha;
          model_t.sv_indices = svIndices_t;
        }
        break;//Switch
      case RANDOM:
        Random random = new Random(alphaDistributionMethodSeed);
        int randomIndex = random.nextInt(alpha.length); // new strategy could be to pick only SV not at bound C
        while(randomIndex == smallestPossibleIndex && alpha.length > 1){ // new assignment if randomIndex is the leftout index
          randomIndex = random.nextInt(alpha.length); // try again
        }
        //give the random point all alpha of the left out point
        double oldAlphaRandomIndex = alpha[randomIndex];
        alpha[randomIndex] += alpha_t;
        if(alpha[randomIndex]>param.C){
          if (LOG.isVerbose()) {LOG.verbose("WARNING at sample " + t + "! over upper bound warning");}
          handleOverBound(param, prob, model_t, t, alpha, svIndices_t, random, randomIndex);//wird danach ueberhaupt der Gradient geupdatet?
        }else {
          model_t.sv_coef[0] = alpha;
          model_t.sv_indices = svIndices_t;
        }

        //update G
        delta_alpha_i = alpha[randomIndex] - oldAlphaRandomIndex;
        double delta_alpha_j = 0 - alpha_t;
        Q_i = svmManager.getQ(svIndices_t[randomIndex]-1,G.length);
        float[] Q_j = svmManager.getQ(t,G.length);
        for(int k=0;k<G.length;k++)
        {
          G[k] += Q_i[k]*delta_alpha_i + Q_j[k]*delta_alpha_j;
        }
        // update G_bar
        if( (oldAlphaRandomIndex >= param.C && alpha[randomIndex] < param.C) || (oldAlphaRandomIndex < param.C && alpha[randomIndex] >= param.C)){ //upper bound changed check: see svm solver update alpha_status and G_bar
          if( oldAlphaRandomIndex >= param.C){ //is upper bound
              for(int j=0;j<G_bar.length;j++){
                  G_bar[j] -= param.C * Q_i[j]; //upper bound * Q_i[j]
              }
          }else{
              for(int j=0;j<G_bar.length;j++){
                  G_bar[j] += param.C * Q_i[j];
              }
          }
        }
        if( alpha_t >= param.C){ //is upper bound
          for(int k=0;k<l;k++){
              G_bar[k] -= param.C * Q_j[k]; //upper bound * Q_i[j]
          }
        }else{
          for(int k=0;k<l;k++){
              G_bar[k] += param.C * Q_j[k];
          }
        }

        break;//Switch
      case PROPORTIONAL:
        proportional(param, prob, svmManager, G, G_bar, l, model_t, t, alpha, svIndices_t, alpha_t,alphaDistributionMethodParameter);            
        break;//Switch
      case EQUALLY:
        proportional(param, prob, svmManager, G, G_bar, l, model_t, t, alpha, svIndices_t, alpha_t,alphaDistributionMethodParameter);
        break;//Switch
      case NEIGHBOR_HIGH_DIM:
        float[] Q_t = svmManager.getQ(t,prob.l);
        int maxIndex = -1;
        float maxQ = Float.NEGATIVE_INFINITY;
        for(int i = 0; i < svIndices_t.length; i++){
          if(Q_t[svIndices_t[i]-1]>maxQ){
            maxQ = Q_t[svIndices_t[i]-1];
            maxIndex = i;
          }
        }
        //give the max point all alpha of the left out point
        double oldAlphaMaxIndex = alpha[maxIndex];
        alpha[maxIndex] += alpha_t;
        if(alpha[maxIndex]>param.C){
          if (LOG.isVerbose()) {LOG.verbose("WARNING at sample " + t + "! over upper bound exception");}
          handleOverBound(param, prob, model_t, t, alpha, svIndices_t, new Random(alphaDistributionMethodSeed), maxIndex);
        }else {
          model_t.sv_coef[0] = alpha;
          model_t.sv_indices = svIndices_t;
        }

        //update G
        delta_alpha_i = alpha[maxIndex] - oldAlphaMaxIndex;
        double delta_alpha_t = 0 - alpha_t;
        Q_i = svmManager.getQ(svIndices_t[maxIndex]-1,G.length);
        Q_t = svmManager.getQ(t,G.length);
        for(int k=0;k<G.length;k++)
        {
          G[k] += Q_i[k]*delta_alpha_i + Q_t[k]*delta_alpha_t;
        }
        // update G_bar
        if( (oldAlphaMaxIndex >= param.C && alpha[maxIndex] < param.C) || (oldAlphaMaxIndex < param.C && alpha[maxIndex] >= param.C)){ //upper bound changed check: see svm solver update alpha_status and G_bar
          if( oldAlphaMaxIndex >= param.C){ //is upper bound
              for(int j=0;j<G_bar.length;j++){
                  G_bar[j] -= param.C * Q_i[j]; //upper bound * Q_i[j]
              }
          }else{
              for(int j=0;j<G_bar.length;j++){
                  G_bar[j] += param.C * Q_i[j];
              }
          }
        }
        if( alpha_t >= param.C){ //is upper bound
          for(int k=0;k<l;k++){
              G_bar[k] -= param.C * Q_t[k]; //upper bound * Q_i[j]
          }
        }else{
          for(int k=0;k<l;k++){
              G_bar[k] += param.C * Q_t[k];
          }
        }
        break;//Switch
    }//endSwitch
    G[t] = G[l-1];
    G_bar[t] = G_bar[l-1];
    svmManager.setG(G);
    svmManager.setG_bar(G_bar);
  }

  /**
   * @param param
   * @param prob
   * @param svmManager
   * @param G
   * @param G_bar
   * @param l
   * @param model_t
   * @param t
   * @param alpha
   * @param svIndices_t
   * @param alpha_t
   */
  private void proportional(svm_parameter param, svm_problem prob, svm svmManager, double[] G, double[] G_bar, int l, svm_model model_t, int t, double[] alpha, int[] svIndices_t, double alpha_t, alphaDistributionMethod alphaDistributionMethodParameter ) {
    float[] Q_i;
    double delta_alpha_i;
    Random random;
    double[] oldAlpha = alpha.clone();
    if(alphaDistributionMethodParameter==alphaDistributionMethod.PROPORTIONAL) {
      balanceProportional(param,alpha,alpha_t,prob.l);
    }else { //Equally
      double delta = alpha_t/alpha.length; //alpha.length needs to be > 0
      for(int i = 0; i < alpha.length; i++) {
        alpha[i] += delta;
      }
    }
    double collectCarry = 0.0;
    List<Integer> notAtBound = new LinkedList<Integer>(); 
    for(int i = 0; i < alpha.length; i++) {
      if( alpha[i] > param.C ) {
        alpha[i] = param.C;
        collectCarry += alpha[i] - param.C;
      }else {
        notAtBound.add(i);
      }
    }
    Iterator<Integer> iter = notAtBound.iterator();
    double diff = 0;
    int indexNotAtBound = -1;
    while(collectCarry >= 0 && iter.hasNext()) {
      indexNotAtBound = iter.next();
      diff = param.C - alpha[indexNotAtBound];
      if( collectCarry >= diff ) {
        alpha[indexNotAtBound] = param.C;
      }else {
        alpha[indexNotAtBound] += diff;
      }
      collectCarry -= diff;
    }
    double[] alphaOldGrow;
    if(collectCarry >= 0 && !iter.hasNext()) {//awake
      int growArrayIndices = (int) Math.ceil(collectCarry / param.C);
      double[] alphaGrow = new double[alpha.length + growArrayIndices];
      int[] svIndicesGrow = new int[alphaGrow.length];
      alphaOldGrow = new double[alphaGrow.length];
      for(int i = 0; i < alpha.length; i++) {
        alphaGrow[i] = alpha[i];
        alphaOldGrow[i] = oldAlpha[i];
        svIndicesGrow[i] = svIndices_t[i];
      }
      random = new Random(alphaDistributionMethodSeed);
      int randomNotSV = random.nextInt(); 
      boolean allPointsAreSV = false;
      for(int i = alpha.length; i < alphaGrow.length - 1 ; i++) {
        alphaGrow[i] = param.C;
        alphaOldGrow[i] = 0;
        int counter = 0;
        while( contains(svIndicesGrow,randomNotSV) || randomNotSV == t || randomNotSV < 1) {
          counter++;
          randomNotSV = random.nextInt();
          if(counter > 1000) {
            break;
          }
        }
        if(counter <= 1000) {
          svIndicesGrow[i] = randomNotSV; //wake up
        }else {
          allPointsAreSV = true;
        }
        
      }
      if(!allPointsAreSV) {
        alphaGrow[alphaGrow.length - 1] = collectCarry % param.C; 
        //bring alpha_old on same length with 0 values
        //svIndices update with random init neq SV
        
        model_t.sv_coef[0] = alphaGrow;
        model_t.sv_indices = svIndicesGrow;
      }else {
        model_t.sv_coef[0] = alpha; // doesn't satisfy KKT Conditions
        model_t.sv_indices = svIndices_t;
        alphaOldGrow = oldAlpha;
      }
    }else {//not awake
      model_t.sv_coef[0] = alpha;
      model_t.sv_indices = svIndices_t;
      alphaOldGrow = oldAlpha;
    }
    
    //swap max sv_indices to last position
    int maxIndices=-1;
    int indexOfMaxIndices = -1;
    for(int i = 0; i < model_t.sv_indices.length; i++) {
      if ( model_t.sv_indices[i] > maxIndices ) {
        maxIndices = model_t.sv_indices[i];
        indexOfMaxIndices = i;
      }  
    }
    model_t.sv_indices[indexOfMaxIndices] = model_t.sv_indices[model_t.sv_indices.length - 1];
    model_t.sv_indices[model_t.sv_indices.length - 1] = maxIndices;
    double tempAlpha = model_t.sv_coef[0][indexOfMaxIndices];
    model_t.sv_coef[0][indexOfMaxIndices] = model_t.sv_coef[0][model_t.sv_indices.length - 1];
    model_t.sv_coef[0][model_t.sv_indices.length - 1] = tempAlpha;
    
    
    for(int i = 0; i < model_t.sv_coef[0].length; i++) {
      //update G with i
      delta_alpha_i = model_t.sv_coef[0][i] - alphaOldGrow[i];
      Q_i = svmManager.getQ(svIndices_t[i] - 1, G.length);
      for (int k = 0; k < l; k++)
      {
          G[k] += Q_i[k] * delta_alpha_i;
      }
      // update G_bar
      if( (alphaOldGrow[i] >= param.C && model_t.sv_coef[0][i] < param.C) || (alphaOldGrow[i] < param.C && model_t.sv_coef[0][i] >= param.C)){ //upper bound changed check: see svm solver update alpha_status and G_bar
        if( oldAlpha[i] >= param.C){ //is upper bound
          for(int j=0;j<l;j++){ 
            G_bar[j] -= param.C * Q_i[j]; //upper bound * Q_i[j]
          }
        }else{
          for(int j=0;j<l;j++){
            G_bar[j] += param.C * Q_i[j];
          }
        }
      }
    }
  }


  /**
   * @param param
   * @param prob
   * @param model_t
   * @param t
   * @param alpha
   * @param svIndices_t
   * @param random
   * @param randomIndex
   * If the leftout alpha distribution leads to overfilling, the carry is distributed to other points.
   */
  private void handleOverBound(svm_parameter param, svm_problem prob, svm_model model_t, int t, double[] alpha, int[] svIndices_t, Random random, int randomIndex) {
    double carry = alpha[randomIndex] - param.C;
    //if new point should be awaken, then old model_t.sv_coef[0] can be used because it has old length
    //model_t.sv_coef[0][randomIndex] = param.C;
    //model_t.sv_coef[0][smallestPossibleIndex] = carry;// change at position smallestPossibleIndex (=removed index)
    int wakened = random.nextInt(prob.l - 1);
    if(prob.l > svIndices_t.length) {
      int counter = 0;
      boolean allPointsAreSV = false;
      while(contains(svIndices_t,wakened) || wakened == t || wakened < 1) {
        counter++; // counts iterations
        wakened = random.nextInt(prob.l-1);
        if(counter>1000) {// break infinity loop (only if no value found for wakened)
          //probably all points are SV and no one can be woken up
          allPointsAreSV = true;
          break;
        }
      }
      if(!allPointsAreSV) {
        int[] svIndices_t_wakened = new int[svIndices_t.length + 1];
        double[] alpha_t_wakened = new double[svIndices_t.length + 1];
        for(int i = 0; i < svIndices_t.length; i++) {
          svIndices_t_wakened[i] = svIndices_t[i];
          alpha_t_wakened[i] = alpha[i];
        }
        if(alpha_t_wakened[randomIndex] >= param.C) {
          alpha_t_wakened[randomIndex] = param.C;
          alpha_t_wakened[svIndices_t.length] = carry; // wakened at last position
        }else {
          alpha_t_wakened[randomIndex] = carry;
          alpha_t_wakened[svIndices_t.length] = param.C;; // wakened at last position
        }
        svIndices_t_wakened[svIndices_t.length] = wakened;
        model_t.sv_indices = svIndices_t_wakened;
        model_t.sv_coef[0] = alpha_t_wakened;
      }else { //all points are SV
        alpha[randomIndex] = param.C;
        model_t.sv_coef[0] = alpha; //problem doesn't satisfy the KKT Condition, because sum alpha < 1
        model_t.sv_indices = svIndices_t;
      }
      //swap max sv_indices to last position
      int maxIndices=-1;
      int indexOfMaxIndices = -1;
      for(int i = 0; i < model_t.sv_indices.length; i++) {
        if ( model_t.sv_indices[i] > maxIndices ) {
          maxIndices = model_t.sv_indices[i];
          indexOfMaxIndices = i;
        }  
      }
      model_t.sv_indices[indexOfMaxIndices] = model_t.sv_indices[model_t.sv_indices.length - 1];
      model_t.sv_indices[model_t.sv_indices.length - 1] = maxIndices;
      double tempAlpha = model_t.sv_coef[0][indexOfMaxIndices];
      model_t.sv_coef[0][indexOfMaxIndices] = model_t.sv_coef[0][model_t.sv_indices.length - 1];
      model_t.sv_coef[0][model_t.sv_indices.length - 1] = tempAlpha;
      
    }else {
      alpha[randomIndex] = param.C;
      model_t.sv_coef[0] = alpha;
      model_t.sv_indices = svIndices_t;
    }
  }
  
  /**
   * checks if j is contained in array 
   */
  private boolean contains(int[] array, int j) {
    for(int i : array) {
      if(i == j) {
        return true;
      }
    }
    return false;
  }

  private int getIndexJ(Relation<V> relation, int dim, ArrayDBIDs ids, svm svmManager, WritableDoubleDataStore scores, nextLeaveMoreOutElementMethod leaveMoreOutMethod, svm_model modelPath, svm_problem probOld, Stack<int[]> swaps, int round) {
    int indexJ = 0;
    switch (leaveMoreOutMethod) {//delete strategies
      case RANDOM://random
        Random random = new Random(nextLeaveMoreOutElementMethodSeed);
        indexJ = random.nextInt(modelPath.sv_indices.length);
        break;
      case MAXSCORE://get SV from current model which has the maximum score
        double maxScore=Double.NEGATIVE_INFINITY;
        for (int i = 0; i < modelPath.sv_indices.length; i++) {          
          int idJ = probOld.id[modelPath.sv_indices[i]-1]; 
          DBIDRef iterJ = ids.get(idJ);
          double scoreJ = scores.doubleValue(iterJ);
          if( scoreJ > maxScore){
            maxScore = scoreJ;
            indexJ = i;
          }
        }
        break;
      case NEIGHBOR_HIGH_DIM:
        indexJ = getMaxQ(probOld,svmManager,modelPath,swaps.lastElement()[1]);// if xIndexOnPath is changed in the cache in round before, then backwartsSwap with use of swaps
        break;
    }
    return indexJ;
  }

  private svm_problem cloneProb(svm_problem prob_t, svm_problem probOld) {
    probOld.l = prob_t.l;
    probOld.x = prob_t.x.clone(); //needs O(l)!
    probOld.y = prob_t.y;
    probOld.id = prob_t.id.clone();
    return probOld;
  }

  private svm_model cloneModel(svm_model model_t) {
    svm_model modelOld = new svm_model();
    modelOld.l = model_t.l;
    modelOld.sv_indices = model_t.sv_indices.clone();
    modelOld.rho = model_t.rho.clone();
    modelOld.sv_coef = model_t.sv_coef.clone();
    modelOld.SV = model_t.SV.clone();
    modelOld.Gradient = model_t.Gradient;
    modelOld.Gradient_bar = model_t.Gradient_bar;
    return modelOld;
  }

  private double getScore(svm_node[] x, int dim, svm_model model, double[] buf, svm_model model_t, int t, int nextSVIndex) {
    // Estimate Score of t-th Sample
    if (LOG.isVerbose()) {LOG.verbose("Predicting the score of the " + t + "-th Sample...");}

    //double norm_w = 0.0;
    if(t != nextSVIndex - 1) {
      svm.svm_predict_values(model_t, x, buf);
      //norm_w = SVMUtils.calculateNorm_w(model_t);
    }else {
      svm.svm_predict_values(model, x, buf);
      //norm_w = SVMUtils.calculateNorm_w(model);
    }
    return -buf[0];// / norm_w;
  }

  /**
   *
   * @param prob for parameter l
   * @param svmManager contains cache
   * @param model
   * @param t left out sample index
   * @return the index of the maximum Q. It should be the most similar point to the left out point t.
   */
  private int getMaxQ(svm_problem prob, svm svmManager, svm_model model, int t) {
    float[] Q_t = svmManager.getQ(t,prob.l);
    int maxIndex = -1;
    float maxQ = Float.NEGATIVE_INFINITY;
    for(int i = 0; i < model.sv_indices.length; i++){
      if(Q_t[model.sv_indices[i]-1]>maxQ){
        maxQ = Q_t[model.sv_indices[i]-1];
        maxIndex = i;
      }
    }
    return maxIndex;
  }


  public enum alphaDistributionMethod {PROPORTIONAL, EQUALLY, NEIGHBOR_HIGH_DIM, RANDOM, RANDOM_NO_SV}
  public enum nextLeaveMoreOutElementMethod {RANDOM, MAXSCORE, NEIGHBOR_HIGH_DIM} 
  public enum aggregateMethod{OVERRIDE, WEIGHTED_AVERAGE} // only if path > 1

  private NextOldLeftOutElement createProb_t_WithoutCopy(svm_problem prob_t, int t, int old_t, svm_node[] old_x, int old_id) {
    // create state that the last element is removed
    svm_node[] lastElement = prob_t.x[old_t]; 
    int lastElementID = prob_t.id[old_t];
    prob_t.x[old_t] = old_x;
    prob_t.id[old_t] = old_id;
    
    // create state that the last element is at the position t
    if(t < prob_t.l) {
      svm_node[] new_x = prob_t.x[t];
      int new_id = prob_t.id[t];
      prob_t.x[t] = lastElement;
      prob_t.id[t] = lastElementID;
      //return new_x,new_id for next t -> introduce class which contain both
      return new NextOldLeftOutElement(new_x,new_id);
    } else {
      return null;
    }
  }
  
  /**
   * Container for return variables because java can not return two variables 
   */
  private class NextOldLeftOutElement{
    private svm_node[] new_x;
    private int new_id;
    public NextOldLeftOutElement(svm_node[] new_x2, int new_id2) {
      this.new_x = new_x2;
      this.new_id = new_id2;
    }
    public svm_node[] getNew_x() {
      return new_x;
    }
    public int getNew_id() {
      return new_id;
    }    
  }
  
  //Create new Problem with t-th Sample removed
  private NextOldLeftOutElement createProb_t(svm_problem prob, svm_problem prob_t, int t) {
    prob_t.l = prob.l - 1;
    prob_t.x = new svm_node[prob.l - 1][];
    prob_t.y = new double[prob.l - 1];
    prob_t.id = new int[prob.l - 1];

    svm_node[] old_x = prob.x[t].clone();
    int old_id = prob.id[t];
    
    for (int i = 0; i < prob.l - 1; i++) {//only till l-1
      if (i != t) {
        prob_t.x[i] = prob.x[i]; 
        prob_t.y[i] = +1;
        prob_t.id[i] = prob.id[i];
      } else {
        prob_t.x[i] = prob.x[prob.l - 1]; // swap the last element to the position t.
        prob_t.y[i] = +1;
        prob_t.id[i] = prob.id[prob.l - 1];
      }
    }

    return new NextOldLeftOutElement(old_x, old_id);
    //Alpha Swap in adaptModelSize()
  }

  /**
   * Deletes the t-th alpha, and distributes it over the remaining alpha propotional
   *
   *
   * alpha[i] = alpha[i] + alpha[t]*alpha[i]/(nu*l-alpha[t]) = alpha[i]*vl / (vl-alpha[t])
   *
   * @param param SVM Parameter
   * @param alpha Array of alpha > 0
   * @param alpha_t alpha of t-th sample
   */
  private void balanceProportional(svm_parameter param, double[] alpha, double alpha_t, int l) {
    double vl = param.nu * l; //TODO Change to param.C 
    if(vl - alpha_t > 0){
      for(int i = 0; i < alpha.length; i++){
        //double old_alpha_i = alpha[i];
        alpha[i] *= vl;
        alpha[i] /= (vl-alpha_t); // distribute alpha_t propotional
      }
    }else{//Dividing through zero forbidden or negativity
      if (LOG.isVerbose()) {LOG.verbose("WARNING! vl - alpha_t <= 0");}
      for(int i = 0; i < alpha.length; i++){
        alpha[i] = 0;
      }
    }

  }

  @Override
  public TypeInformation[] getInputTypeRestriction() {
    return TypeUtil.array(TypeUtil.NUMBER_VECTOR_FIELD);
  }

  /**
   * Setup logging helper for SVM.
   */
  static final svm_print_interface LOG_HELPER = new svm_print_interface() {
    @Override
    public void print(String arg0) {
      if(LOG.isVerbose()) {
        LOG.verbose(arg0);
      }
    }
  };

  /**
   * Parameterization class.
   *
   * @author Erich Schubert
   *
   * @hidden
   *
   * @param <V> Vector type
   */
  public static class Par<V extends NumberVector> implements Parameterizer {
    /**
     * Parameter for SVM type.
     */
    public static final OptionID TYPE_ID = new OptionID("svm.type", "SVM type parameter.");
    
    /**
     * Parameter for kernel function.
     */
    public static final OptionID KERNEL_ID = new OptionID("svm.kernel", "Kernel to use with SVM.");

    /**
     * OCSVM nu parameter
     */
    public static final OptionID NU_ID = new OptionID("svm.nu", "SVM nu parameter.");
    
    /**
     * SVDD C parameter
     */
    public static final OptionID C_ID = new OptionID("svm.C", "SVM C parameter. For type SVDD: C > 1/(#instances-PathDepth)");
    
    /**
     * SVM gamma parameter
     */
    public static final OptionID GAMMA_ID = new OptionID("svm.gamma", "SVM gamma parameter.");
    
    /**
     * SVM_LOO pathdepth parameter
     */
    public static final OptionID PATHDEPTH_ID = new OptionID("svm.pathdepth", "SVM pathdepth parameter. Greater 1 activates further pathdepth options.");

    /**
     * SVM_LOO zeta parameter
     */
    public static final OptionID ZETA_ID = new OptionID("svm.zeta", "SVM zeta parameter.");
    
    /**
     * SVM_LOO alphaDistributionMethod parameter
     */
    public static final OptionID ALPHA_DISTRIBUTION_METHOD_ID = new OptionID("svm.alphaDistributionMethod", "SVM alphaDistributionMethod parameter.");
    
    
    /**
     * SVM_LOO nextLeaveMoreOutElementMethod parameter
     */
    public static final OptionID NEXT_LEAVE_MORE_OUT_ELEMENT_METHOD_ID = new OptionID("svm.nextLeaveMoreOutElementMethod", "SVM nextLeaveMoreOutElementMethod parameter.");
    
    /**
     * SVM_LOO aggregateMethod parameter
     */
    public static final OptionID AGGREGATE_METHOD_ID = new OptionID("svm.aggregateMethod", "SVM aggregateMethod parameter.");
    
    /**
     * SVM_LOO alphaDistributionMethodSeed parameter
     */
    public static final OptionID ALPHA_DISTRIBUTION_METHOD_SEED_ID = new OptionID("svm.alphaDistributionMethodSeed", "SVM alphaDistributionMethod seed.");
    
    /**
     * SVM_LOO nextLeaveMoreOutElementMethod seed
     */
    public static final OptionID NEXT_LEAVE_MORE_OUT_ELEMENT_METHOD_SEED_ID = new OptionID("svm.nextLeaveMoreOutElementMethodSeed", "SVM nextLeaveMoreOutElementMethod seed.");
    
    /**
     * SVM_LOO randomSampling parameter
     */
    public static final OptionID RANDOM_SAMPLING_ID = new OptionID("svm.randomSampling", "SVM random sampling.");
    
    /**
     * SVM_LOO randomSampling seed
     */
    public static final OptionID RANDOM_SAMPLING_SEED_ID = new OptionID("svm.randomSamplingSeed", "SVM random sampling seed.");
    
    /**
     * SVM_LOO alternateRandomSampling parameter
     */
    public static final OptionID ALTERNATE_RANDOM_SAMPLING_ID = new OptionID("svm.AlternateRandomSampling", "SVM alternate random sampling.");
    
    /**
     * SVM_LOO randomSamplingDepth parameter
     */
    public static final OptionID RANDOM_SAMPLING_DEPTH_ID = new OptionID("svm.randomSamplingDepth", "SVM random sampling depth.");
    
    /**
     * Type in use.
     */
    protected SVMType type = SVMType.ONE_CLASS;
    
    /**
     * Kernel in use.
     */
    protected SVMKernel kernel = SVMKernel.RBF;

    /**
     * Nu parameter.
     */
    protected double nu = 0.05;
    
    /**
     * C parameter.
     */
    protected double C = 1.0;

    /**
     * Gamma parameter.
     */
    protected double gamma = 0.5;
    
    /**
     * Pathdepth parameter.
     */
    protected int pathdepth = 1;
    
    /**
     * nextLeaveMoreOutElementMethod in use.
     */
    protected nextLeaveMoreOutElementMethod nextLeaveMoreOutElementMethodParameter = nextLeaveMoreOutElementMethod.RANDOM;
    
    /**
     * nextLeaveMoreOutElementMethodSeed seed.
     */
    protected Long nextLeaveMoreOutElementMethodSeed = Long.parseLong("5432");
    
    /**
     * alphaDistributionMethod in use.
     */
    protected alphaDistributionMethod alphaDistributionMethodParameter = alphaDistributionMethod.RANDOM;
    
    /**
     * alphaDistributionMethod seed.
     */
    protected Long alphaDistributionMethodSeed = Long.parseLong("4321");
    
    /**
     * aggregateMethod in use.
     */
    protected aggregateMethod aggregateMethodParameter = aggregateMethod.WEIGHTED_AVERAGE;
    
    /**
     * Zeta parameter.
     */
    protected double zeta = 0.7;
    
    /**
     * activate random sampling.
     */
    protected boolean randomSampling = false;
    
    /**
     * randomSampling seed.
     */
    protected long randomSamplingSeed = Long.parseLong("4567"); 
    
    /**
     * activate the alternating strategy that alternates between random sampling and nextLeaveMoreOutStrategy.
     */
    protected boolean alternateRandomSampling = true;
    
    /**
     * randomSamplingDepth parameter.
     */
    protected int countRandomSamplingDepth = 5;

    @Override
    public void configure(Parameterization config) {

      EnumParameter<SVMType> typeP = new EnumParameter<>(TYPE_ID, SVMType.class, SVMType.ONE_CLASS);
      if(config.grab(typeP)) {
        type = typeP.getValue();
      }
      
      EnumParameter<SVMKernel> kernelP = new EnumParameter<>(KERNEL_ID, SVMKernel.class, SVMKernel.RBF);
      if(config.grab(kernelP)) {
        kernel = kernelP.getValue();
      }

      if(type == SVMType.ONE_CLASS) {
        DoubleParameter nuP = new DoubleParameter(NU_ID, 0.05);
        nuP.addConstraint(new GreaterConstraint(0));
        nuP.addConstraint(new LessConstraint(1));
        if(config.grab(nuP)) {
          nu = nuP.doubleValue();
        }
      } else {
        DoubleParameter CP = new DoubleParameter(C_ID, 1.0);
        CP.addConstraint(new GreaterConstraint(0));
        CP.addConstraint(new LessEqualConstraint(1));
        if(config.grab(CP)) {
          C = CP.doubleValue();
        }
      }
      
      DoubleParameter gammaP = new DoubleParameter(GAMMA_ID, 0.5);
      if(config.grab(gammaP)) {
        gamma = gammaP.doubleValue();
      }
      
      IntParameter pathDepthP = new IntParameter(PATHDEPTH_ID, 1);
      pathDepthP.addConstraint(new GreaterConstraint(0));
      if(config.grab(pathDepthP)) {
        pathdepth = pathDepthP.getValue();
      }
      
      EnumParameter<alphaDistributionMethod> alphaDistributionMethodP = new EnumParameter<>(ALPHA_DISTRIBUTION_METHOD_ID, alphaDistributionMethod.class, alphaDistributionMethod.RANDOM);
      if(config.grab(alphaDistributionMethodP)) {
        alphaDistributionMethodParameter = alphaDistributionMethodP.getValue();
      }
      
      LongParameter alphaDistributionMethodSeedP = new LongParameter(ALPHA_DISTRIBUTION_METHOD_SEED_ID, Long.parseLong("4321"));
      if(config.grab(alphaDistributionMethodSeedP)) {
        alphaDistributionMethodSeed = alphaDistributionMethodSeedP.getValue();
      }
      
      if(pathdepth > 1) {
        EnumParameter<aggregateMethod> aggregateMethodP = new EnumParameter<>(AGGREGATE_METHOD_ID, aggregateMethod.class, aggregateMethod.WEIGHTED_AVERAGE);
        if(config.grab(aggregateMethodP)) {
          aggregateMethodParameter = aggregateMethodP.getValue();
        }
        
        EnumParameter<nextLeaveMoreOutElementMethod> nextLeaveMoreOutElementMethodP = new EnumParameter<>(NEXT_LEAVE_MORE_OUT_ELEMENT_METHOD_ID, nextLeaveMoreOutElementMethod.class, nextLeaveMoreOutElementMethod.RANDOM);
        if(config.grab(nextLeaveMoreOutElementMethodP)) {
          nextLeaveMoreOutElementMethodParameter = nextLeaveMoreOutElementMethodP.getValue();
        }
        
        DoubleParameter zetaP = new DoubleParameter(ZETA_ID, 0.7);
        zetaP.addConstraint(new LessEqualConstraint(1));
        zetaP.addConstraint(new GreaterEqualConstraint(0));
        if(config.grab(zetaP)) {
          zeta = zetaP.doubleValue();
        }
        
        LongParameter nextLeaveMoreOutElementMethodSeedP = new LongParameter(NEXT_LEAVE_MORE_OUT_ELEMENT_METHOD_SEED_ID, Long.parseLong("5432"));
        if(config.grab(nextLeaveMoreOutElementMethodSeedP)) {
          nextLeaveMoreOutElementMethodSeed = nextLeaveMoreOutElementMethodSeedP.getValue();
        }
        
        Flag randomSamplingP = new Flag(RANDOM_SAMPLING_ID);
        if(config.grab(randomSamplingP)) {
          randomSampling = randomSamplingP.getValue();
        }
        
        if(randomSampling) {      
          LongParameter randomSamplingSeedP = new LongParameter(RANDOM_SAMPLING_SEED_ID, Long.parseLong("4567"));
          if(config.grab(randomSamplingSeedP)) {
            randomSamplingSeed = nextLeaveMoreOutElementMethodSeedP.getValue();
          }
        
          Flag alternateRandomSamplingP = new Flag(ALTERNATE_RANDOM_SAMPLING_ID);
          if(config.grab(alternateRandomSamplingP)) {
            alternateRandomSampling = alternateRandomSamplingP.getValue();
          }
          
          IntParameter randomSamplingDepthP = new IntParameter(RANDOM_SAMPLING_DEPTH_ID, 1);
          randomSamplingDepthP.addConstraint(new GreaterConstraint(0));
          if(config.grab(randomSamplingDepthP)) {
            countRandomSamplingDepth = randomSamplingDepthP.getValue();
          }
        }
      }
    }

   
    @Override
    public Object make() {
      return new LeaveMoreOutLibSVMOneClassOutlierDetection<>(type, kernel, nu, C, gamma, pathdepth, alphaDistributionMethodParameter, aggregateMethodParameter, nextLeaveMoreOutElementMethodParameter, zeta, alphaDistributionMethodSeed, nextLeaveMoreOutElementMethodSeed, randomSampling, randomSamplingSeed, alternateRandomSampling, countRandomSamplingDepth);
    }
    
  }
}
