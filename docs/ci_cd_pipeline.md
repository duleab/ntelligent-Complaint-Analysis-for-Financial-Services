# CI/CD Pipeline Documentation

This document provides detailed information about the Continuous Integration and Continuous Deployment (CI/CD) pipeline implemented for the Intelligent Complaint Analysis System.

## Overview

The CI/CD pipeline automates the testing, building, and deployment processes, ensuring that code changes are thoroughly tested before being deployed to production. The pipeline is implemented using GitHub Actions and is defined in the `.github/workflows/ci_cd.yml` file.

## Pipeline Stages

The pipeline consists of four main stages:

### 1. Test Stage

This stage runs on every push to the `main`, `master`, and `develop` branches, as well as on pull requests to the `main` and `master` branches.

**Steps:**
- Checkout the code
- Set up Python (versions 3.8 and 3.9)
- Install dependencies
- Run linting with flake8
- Execute tests with pytest and generate coverage reports

### 2. Data Pipeline Stage

This stage runs after successful tests, but only on pushes to the `main` or `master` branches.

**Steps:**
- Checkout the code
- Set up Python 3.8
- Install dependencies
- Create necessary directories
- Run the data pipeline with a small sample size
- Cache the vector store for later use

### 3. Build Stage

This stage runs after successful tests and data pipeline, but only on pushes to the `main` or `master` branches.

**Steps:**
- Checkout the code
- Set up Python 3.8
- Install dependencies
- Build the Python package
- Store build artifacts

### 4. Deploy Stage

This stage runs after a successful build, but only on pushes to the `main` or `master` branches.

**Steps:**
- Checkout the code
- Set up Python 3.8
- Install dependencies
- Download build artifacts
- Restore vector store from cache
- Create necessary directories
- Deploy to Streamlit Cloud

## Setting Up Deployment

### Streamlit Cloud Deployment

To enable automatic deployment to Streamlit Cloud:

1. Create a Streamlit Cloud account at [https://streamlit.io/cloud](https://streamlit.io/cloud)
2. Generate an API token from your Streamlit Cloud account
3. Add the token as a repository secret named `STREAMLIT_API_TOKEN` in your GitHub repository settings:
   - Go to your repository on GitHub
   - Click on "Settings"
   - Click on "Secrets and variables" > "Actions"
   - Click on "New repository secret"
   - Name: `STREAMLIT_API_TOKEN`
   - Value: Your Streamlit Cloud API token
   - Click "Add secret"

## Customizing the Pipeline

### Changing Python Versions

To change the Python versions used for testing, modify the `matrix.python-version` array in the `test` job:

```yaml
strategy:
  matrix:
    python-version: [3.8, 3.9]  # Add or remove versions as needed
```

### Changing the Sample Size for Data Processing

To change the sample size used in the data pipeline stage, modify the `--sample-size` parameter in the "Download and process data" step:

```yaml
python run_pipeline.py --sample-size 1000 --vector-store-type faiss
```

### Changing the Vector Store Type

To change the vector store type used in the data pipeline stage, modify the `--vector-store-type` parameter in the "Download and process data" step:

```yaml
python run_pipeline.py --sample-size 1000 --vector-store-type chroma  # Change to "chroma" or "faiss"
```

### Adding Custom Deployment Steps

To add custom deployment steps, modify the "Deploy to Streamlit Cloud" step in the `deploy` job. For example, to add environment variables or configure additional settings:

```yaml
- name: Deploy to Streamlit Cloud
  env:
    STREAMLIT_API_TOKEN: ${{ secrets.STREAMLIT_API_TOKEN }}
    CUSTOM_ENV_VAR: ${{ secrets.CUSTOM_ENV_VAR }}  # Add custom environment variables
  run: |
    # Your custom deployment steps
```

## Troubleshooting

### Pipeline Failures

If the pipeline fails, check the GitHub Actions logs for detailed error messages. Common issues include:

- **Test failures**: Fix the failing tests and push the changes
- **Linting errors**: Fix the linting issues and push the changes
- **Dependency issues**: Update the requirements.txt file if needed
- **Deployment failures**: Check that the `STREAMLIT_API_TOKEN` secret is correctly set

### Missing Vector Store

If the deployment fails due to a missing vector store, ensure that the data pipeline stage is completing successfully and that the vector store is being correctly cached and restored.

## Best Practices

1. **Run tests locally** before pushing changes to avoid pipeline failures
2. **Keep the sample size small** in the data pipeline stage to reduce build times
3. **Monitor pipeline execution times** and optimize steps if needed
4. **Regularly update dependencies** to ensure security and compatibility
5. **Review pipeline logs** to identify and fix issues promptly

## Future Improvements

- Add automated performance testing
- Implement canary deployments
- Add notification systems for pipeline status
- Implement automated rollback on deployment failures
- Add more comprehensive test coverage reporting