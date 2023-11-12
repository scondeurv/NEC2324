using BackPropagation.Configuration;
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
        

        RuleFor(tp => tp.Layers)
            .GreaterThan(0);

        RuleFor(tp => tp.TrainingDataPercentage)
            .Must((tp, p) => (tp.TestDataPercentage + p) < 100);

        RuleFor(tp => tp.UnitsPerLayer.Length)
            .Must((p, l) => l == p.Layers);

        RuleFor(tp => tp.Momentum)
            .GreaterThanOrEqualTo(0.0f);

        RuleFor(tp => tp.DataFile)
            .NotEmpty();
    }
}