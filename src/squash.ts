import Synaptic = require('./synaptic');

// squashing functions

export function LOGISTIC(x: number, derivate?: boolean) {
	if (derivate) {
		var fx = LOGISTIC(x);
		return fx * (1 - fx);
	}
	return 1 / (1 + Math.exp(-x));

}

export function TANH(x: number, derivate?: boolean) {
	if (derivate)
		return 1 - Math.pow(TANH(x), 2);
	var eP = Math.exp(x);
	var eN = 1 / eP;
	return (eP - eN) / (eP + eN);
}

export function IDENTITY(x: number, derivate?: boolean) {
	return derivate ? 1 : x;
}

export function HLIM(x: number, derivate?: boolean) {
	return derivate ? 1 : +(x > 0);
}

export function SOFTPLUS(x: number, derivate?: boolean) {
	if (derivate)
		return 1 - 1 / (1 + Math.exp(x));
	return Math.log(1 + Math.exp(x));
}

export function EXP(x: number, derivate?: boolean) {
	return Math.exp(x);
} 