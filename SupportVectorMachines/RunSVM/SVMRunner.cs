using System.Diagnostics.CodeAnalysis;
using Accord.Statistics.Analysis;
using LibSVMsharp;

namespace SupportVectorMachines.RunSVM;

public sealed class SVMRunner
{
    public (SVMParameter, SVMModel, ConfusionMatrix) RunSVC(
        SVMType svmType,
        SVMProblem problem,
        SVMProblem testProblem,
        double fScoreTarget = 0.9,
        SVMKernelType kernelType = SVMKernelType.POLY,
        SVMOptimizer optimizer = SVMOptimizer.RandomSearch,
        int iterations = 100)
    {
        SVMModel bestModel = null;
        SVMParameter bestParameter = null;
        ConfusionMatrix confusionMatrix = null;

        const double csStart = 0.1;
        const int csEnd = 100;
        const double csStep = 0.1;
        const double nuStart = 0.01;
        const double nuEnd = 1.0;
        const double nuStep = 0.01;
        const double gammaStart = 0.1;
        const int gammaEnd = 100;
        const double gammaStep = 0.1;
        
        switch (optimizer)
        {
            case SVMOptimizer.GridSearch:
                (bestParameter, bestModel, confusionMatrix) = GridSearch(svmType, problem, testProblem, Run, svmType == SVMType.C_SVC ? csStart : nuStart,
                    svmType == SVMType.C_SVC ? csEnd : nuEnd, svmType == SVMType.C_SVC ? csStep : nuStep, gammaStart, gammaEnd, gammaStep, fScoreTarget, kernelType);
                break;
            case SVMOptimizer.RandomSearch:
                (bestParameter, bestModel, confusionMatrix) = RandomSearch(svmType, problem, testProblem, Run, svmType == SVMType.C_SVC ? csStart : nuStart,
                    svmType == SVMType.C_SVC ? csEnd : nuEnd, svmType == SVMType.C_SVC ? csStep : nuStep, gammaStart, gammaEnd, gammaStep, fScoreTarget, kernelType, iterations);
                break;
        }

        return (bestParameter, bestModel, confusionMatrix);
    }

    
    private static (SVMParameter, SVMModel, ConfusionMatrix) Run(SVMType svmType, SVMProblem problem, SVMProblem testProblem, double cOrNu,
        double gamma, int degree, SVMKernelType kernel)
    {
        var parameter = new SVMParameter
        {
            Type = svmType,
            Kernel = kernel,
            Degree = degree,
            Gamma = gamma,
        };

        switch (svmType)
        {
            case SVMType.C_SVC:
                parameter.C = cOrNu;
                break;
            case SVMType.NU_SVC:
                parameter.Nu = cOrNu;
                break;
            default:
                throw new NotSupportedException(nameof(svmType));
        }
        
        var model = SVM.Train(problem, parameter);
        var cm = Predict(model, testProblem).ConfusionMatrix;
        
        return (parameter, model, cm);
    }
    
    public static (ConfusionMatrix ConfusionMatrix, int[] Expected, int[] Predicted) Predict(SVMModel model, SVMProblem testProblem)
    {
        var target = new double[testProblem.Length];
        SVM.CrossValidation(testProblem, model.Parameter, 5, out target);

        var expected = Array.ConvertAll(testProblem.Y.ToArray(), x => (int)x);
        var predicted = Array.ConvertAll(target, x => (int)x);
        var cm = new ConfusionMatrix(predicted, expected);
        return (cm, expected, predicted);
    }
    
    private static (SVMParameter, SVMModel, ConfusionMatrix) GridSearch(SVMType smvType, SVMProblem problem,
        SVMProblem testProblem,
        Func<SVMType, SVMProblem, SVMProblem, double, double, int, SVMKernelType, (SVMParameter, SVMModel, ConfusionMatrix)>
            run,
        double cOrNusStart = 0.1,
        double csOrNusEnd = 100,
        double csOrNusStep = 0.1,
        double gammaStart = 0.1,
        double gammaEnd = 100,
        double gammaStep = 0.1,
        double fScoreTarget = 0.9,
        SVMKernelType kernelType = SVMKernelType.POLY)
    {
        double bestFScore = 0;
        SVMModel bestModel = null;
        SVMParameter bestParameter = null;
        ConfusionMatrix bestConfusionMatrix = null;

        var csOrNus = GenerateRange(cOrNusStart, csOrNusEnd, csOrNusStep);
        var gammas = GenerateRange(gammaStart, gammaEnd, gammaStep);
        var degrees = Enumerable.Range(2, 4).Select(i => i).ToArray();

        for (var i = 0; i < csOrNus.Length; i++)
        {
            for (var j = 0; j < gammas.Length; j++)
            {
                for (var k = 0; k < degrees.Length; k++)
                {
                    var (parameter, model, confusionMatrix) = run(smvType, problem, testProblem, csOrNus[i], gammas[j], degrees[k],
                        kernelType);
                    if (confusionMatrix.FScore > bestFScore)
                    {
                        bestFScore = confusionMatrix.FScore;
                        bestParameter = parameter;
                        bestModel = model;
                        bestConfusionMatrix = confusionMatrix;
                    }

                    if (bestFScore >= fScoreTarget)
                    {
                        break;
                    }
                }
            }
        }

        return (bestParameter, bestModel, bestConfusionMatrix);
    }

    private static (SVMParameter, SVMModel, ConfusionMatrix) RandomSearch(SVMType svmType, SVMProblem problem,
        SVMProblem testProblem,
        Func<SVMType, SVMProblem, SVMProblem, double, double, int, SVMKernelType, (SVMParameter, SVMModel, ConfusionMatrix)>
            run,
        double cOrNusStart = 0.1,
        double csOrNusEnd = 100,
        double csOrNusStep = 0.1,
        double gammaStart = 0.1,
        double gammaEnd = 100,
        double gammaStep = 0.1,
        double fScoreTarget = 0.9,
        SVMKernelType kernelType = SVMKernelType.POLY,
        int iterations = 100)
    {
        double bestFScore = 0;
        SVMModel bestModel = null;
        SVMParameter bestParameter = null;
        ConfusionMatrix bestConfusionMatrix = null;
        
        var cs = GenerateRange(cOrNusStart, csOrNusEnd, csOrNusStep);
        var gammas = GenerateRange(gammaStart, gammaEnd, gammaStep);
        var degrees = Enumerable.Range(2, 4).Select(i => i).ToArray();
        var rand = new Random();

        for (var i = 0; i < iterations; i++)
        {
            var c = cs[rand.Next(cs.Length)];
            var gamma = gammas[rand.Next(gammas.Length)];
            var degree = degrees[rand.Next(degrees.Length)];
            var (parameter, model, confusionMatrix) =
                run(svmType, problem, testProblem, c, gamma, degree, kernelType);
            if (confusionMatrix.FScore > bestFScore)
            {
                bestFScore = confusionMatrix.FScore;
                bestParameter = parameter;
                bestModel = model;
                bestConfusionMatrix = confusionMatrix;
            }

            if (bestFScore >= fScoreTarget)
            {
                break;
            }
        }

        return (bestParameter, bestModel, bestConfusionMatrix);
    }

    private static double[] GenerateRange(double start, double end, double step)
    {
        var list = new List<double>();
        for (var i = start; i <= end; i += step)
        {
            list.Add(i);
        }

        return list.ToArray();
    }

    public enum SVMOptimizer
    {
        GridSearch,
        RandomSearch,
    }
}