#pragma once
#include <iostream>
#include <vector>
#include "Helper.h"
#include "Urysohn.h"
#include "Layer.h"

class KANKAN {
public:
	KANKAN(const std::vector<int>& U, const std::vector<int>& P, const std::vector<double>& argmin,
		const std::vector<double>& argmax, const std::vector<double>& alphas) {
		if (U.size() != P.size()) {
			printf("Fatal: configuration error 1");
			exit(0);
		}
		if (argmin.size() != argmax.size()) {
			printf("Fatal: configuration error 2");
			exit(0);
		}
		int nLayers = (int)P.size();
		int nFeatures = (int)argmin.size();
		_layers.push_back(std::move(std::make_unique<Layer>(U[0], nFeatures, argmin, argmax, P[0])));
		for (int k = 1; k < nLayers; ++k) {
			_layers.push_back(std::move(std::make_unique<Layer>(U[k], U[k - 1], P[k])));
		}
		for (int k = 0; k < nLayers; ++k) {
			_models.push_back(std::move(std::make_unique<double[]>(U[k])));
			_deltas.push_back(std::move(std::make_unique<double[]>(U[k])));
			_alphas.push_back(alphas[k]);
			_U.push_back(U[k]);
		}
		auto derivatives0 = std::make_unique<std::unique_ptr<double[]>[]>(U[0]);
		for (int i = 0; i < U[0]; ++i) {
			derivatives0[i] = std::make_unique<double[]>(nFeatures);
		}
		_derivatives.push_back(std::move(derivatives0));
		for (int k = 1; k < nLayers; ++k) {
			auto derivativesOther = std::make_unique<std::unique_ptr<double[]>[]>(U[k]);
			for (int i = 0; i < U[k]; ++i) {
				derivativesOther[i] = std::make_unique<double[]>(U[k - 1]);
			}
			_derivatives.push_back(std::move(derivativesOther));
		}
	}
	void Train(const std::unique_ptr<double[]>& features, const std::unique_ptr<double[]>& targets) {
		int nLast = (int)_layers.size() - 1;
		DeepCompute(features, _models[nLast]);
		for (int j = 0; j < _U[nLast]; ++j) {
			_deltas[nLast][j] = targets[j] - _models[nLast][j];
		}
		ComputeDeltas(_deltas[nLast]);
		Update(features);
	}
	void Predict(const std::unique_ptr<double[]>& input, std::unique_ptr<double[]>& output) {
		_layers[0]->Input2Output(input, _models[0]);
		for (int k = 1; k < _layers.size() - 1; ++k) {
			_layers[k]->Input2Output(_models[k - 1], _models[k]);
		}
		int nLast = (int)_layers.size() - 1;
		_layers[nLast]->Input2Output(_models[nLast - 1], output);
	}
private:
	std::vector<std::unique_ptr<Layer>> _layers;
	std::vector<std::unique_ptr<double[]>> _models;
	std::vector<std::unique_ptr<double[]>> _deltas;
	std::vector<double> _alphas;
	std::vector<int> _U;
	std::vector<std::unique_ptr<std::unique_ptr<double[]>[]>> _derivatives;
	//
	void DeepCompute(const std::unique_ptr<double[]>& input, std::unique_ptr<double[]>& output) {
		_layers[0]->Input2Output(input, _models[0], _derivatives[0]);
		for (int k = 1; k < _layers.size() - 1; ++k) {
			_layers[k]->Input2Output(_models[k - 1], _models[k], _derivatives[k]);
		}
		int nLast = (int)_layers.size() - 1;
		_layers[nLast]->Input2Output(_models[nLast - 1], output, _derivatives[nLast]);

	}
	void ComputeDeltas(const std::unique_ptr<double[]>& deltas) {
		int nLast = (int)_layers.size() - 1;
		for (int k = 0; k < _U[nLast]; ++k) {
			_deltas[nLast][k] = deltas[k];
		}
		for (int k = (int)_layers.size() - 1; k >= 1; --k) {
			_layers[k]->ComputeDeltas(_derivatives[k], _deltas[k], _deltas[k - 1], _U[k - 1], _U[k]);
		}
	}
	void Update(const std::unique_ptr<double[]>& input) {
		_layers[0]->Update(input, _deltas[0], _alphas[0]);
		for (int k = 1; k < _layers.size(); ++k) {
			_layers[k]->Update(_models[k - 1], _deltas[k], _alphas[k]);
		}
	}
};
