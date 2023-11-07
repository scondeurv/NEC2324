using BackPropagation.Configuration;
using BackPropagation.Scaling;
using FluentValidation;

namespace BackPropagation.Validation;

public class TrainingParametersValidator : AbstractValidator<TrainingParameters>
{
    public TrainingParametersValidator()
    {
        RuleFor(tp => tp.LearningRate)
            .GreaterThan(0.0f);

        RuleFor(tp => tp.Epochs)
            .GreaterThan(0);

        RuleFor(tp => tp.OutputFile)
            .NotEmpty();

        RuleFor(tp => tp.ActivationFunction)
            .Must(s => Enum.TryParse(typeof(ActivationFunction), (string?)s, true, out var result));
        
        RuleFor(tp => tp.Layers)
            .GreaterThan(0);

        RuleFor(tp => tp.TestDataPercentage)
            .GreaterThan(0);
        
        RuleFor(tp => tp.TrainingDataPercentage)
            .GreaterThan(0);
        
        RuleFor(tp => tp.UnitPerLayer)
            .GreaterThan(0);

        RuleFor(tp => tp.Momentum)
            .GreaterThan(0.0f);
        
        RuleFor(tp => tp.DataFile)
            .NotEmpty();
    }
}