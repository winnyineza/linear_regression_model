import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';

class PredictionScreen extends StatefulWidget {
  const PredictionScreen({super.key});

  @override
  State<PredictionScreen> createState() => _PredictionScreenState();
}

class _PredictionScreenState extends State<PredictionScreen> {
  final _formKey = GlobalKey<FormState>();
  final _yearController = TextEditingController();
  String _prediction = '';
  bool _isLoading = false;

  Future<void> _getPrediction() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() {
      _isLoading = true;
      _prediction = '';
    });

    try {
      final response = await http.post(
        Uri.parse('YOUR_API_ENDPOINT_HERE/predict'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'year': int.parse(_yearController.text)}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        setState(() {
          _prediction =
              'Predicted Temperature Anomaly: ${data['predicted_anomaly'].toStringAsFixed(2)}°C';
        });
      } else {
        setState(() {
          _prediction = 'Error: Failed to get prediction';
        });
      }
    } catch (e) {
      setState(() {
        _prediction = 'Error: $e';
      });
    } finally {
      setState(() {
        _isLoading = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Temperature Anomaly Predictor'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.stretch,
            children: [
              TextFormField(
                controller: _yearController,
                decoration: const InputDecoration(
                  labelText: 'Enter Year (1880-2100)',
                  border: OutlineInputBorder(),
                ),
                keyboardType: TextInputType.number,
                validator: (value) {
                  if (value == null || value.isEmpty) {
                    return 'Please enter a year';
                  }
                  final year = int.tryParse(value);
                  if (year == null) {
                    return 'Please enter a valid year';
                  }
                  if (year < 1880 || year > 2100) {
                    return 'Year must be between 1880 and 2100';
                  }
                  return null;
                },
              ),
              const SizedBox(height: 16),
              ElevatedButton(
                onPressed: _isLoading ? null : _getPrediction,
                child: _isLoading
                    ? const CircularProgressIndicator()
                    : const Text('Predict'),
              ),
              const SizedBox(height: 16),
              if (_prediction.isNotEmpty)
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Text(
                      _prediction,
                      style: const TextStyle(fontSize: 18),
                      textAlign: TextAlign.center,
                    ),
                  ),
                ),
            ],
          ),
        ),
      ),
    );
  }

  @override
  void dispose() {
    _yearController.dispose();
    super.dispose();
  }
}
