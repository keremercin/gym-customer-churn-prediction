# Synthetic Churn Baseline Demo

## Problem
This repository demonstrates a classification workflow on a generated sample dataset. It is not based on a gym operator or observed customer behavior.

## Solution
A machine learning pipeline that:
- preprocesses mixed numerical/categorical data,
- compares multiple models,
- serves an uncalibrated model score and class prediction through a local API.

The generator assigns labels from a rule based on monthly contract, low visits, support-ticket count, and short tenure, with added random noise. These labels are synthetic and do not establish real retention impact. Model selection and reporting use the same single stratified validation split. No calibration, fairness review, or external validation is included.

This is not a customer case study, and the endpoint is not production-ready. The repository does not state a separate data license for the generated CSV.
