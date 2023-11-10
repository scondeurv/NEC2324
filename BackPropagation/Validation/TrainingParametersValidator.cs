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

        //RuleFor(tp => tp.ActivationFunction)
        //    .Must(s => Enum.TryParse(typeof(ActivationFunctionType), (string?)s, true, out var result));

        RuleFor(tp => tp.Layers)
            .GreaterThan(0);

        RuleFor(tp => tp.TrainingDataPercentage)
            .InclusiveBetween(0, 100);

        RuleFor(tp => tp.UnitsPerLayer.Length)
            .Must((p, l) => l == p.Layers);

        RuleFor(tp => tp.Momentum)
            .GreaterThan(0.0f);

        RuleFor(tp => tp.DataFile)
            .NotEmpty();
    }
}