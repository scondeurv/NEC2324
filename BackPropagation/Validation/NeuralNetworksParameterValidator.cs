using BackPropagation.Configuration;
using FluentValidation;

namespace BackPropagation.Validation;

public class NeuralNetworksParameterValidator : AbstractValidator<NeuralNetworkParameters>
{
    public NeuralNetworksParameterValidator()
    {
        RuleFor(tp => tp.LearningRate)
            .GreaterThan(0.0f);

        RuleFor(tp => tp.Epochs)
            .GreaterThan(0);
        
        RuleFor(tp => tp.Layers)
            .GreaterThan(0);

        RuleFor(tp => tp.ValidationPercentage)
            .InclusiveBetween(0, 50);

        RuleFor(tp => tp.UnitsPerLayer.Length)
            .Must((p, l) => l == p.Layers)
            .WithMessage("The number of units per layer must be provided");
        
        RuleFor(tp => tp.UnitsPerLayer)
            .Must((p, u) => u[^1] == 1)
            .WithMessage("Only one output is allowed");

        RuleFor(tp => tp.Momentum)
            .GreaterThanOrEqualTo(0.0f);

        RuleFor(tp => tp.DataFile)
            .NotEmpty();
    }
}