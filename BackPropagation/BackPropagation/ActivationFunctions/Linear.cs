﻿namespace BackPropagation.ActivationFunctions;

public sealed class Linear : IActivationFunction
{
    public double Eval(double input) => input;
    public double Derivative(double input) => 1;
}