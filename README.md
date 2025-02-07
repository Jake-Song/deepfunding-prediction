# Deepfunding Prediction Challenge
Implementation of the V-Index, which is a measure of the open source project’s impact in the ecosystem.

## Quickstart

Make sure you have [`uv` installed](https://docs.astral.sh/uv/). Then run the following command to install the dependencies.

```bash
uv sync
```

### Environment

Create a `.env` file in the root directory with the following variables:

- `GOOGLE_APPLICATION_CREDENTIALS`: Path to a Google Cloud service account with BigQuery access. You can create one on the [Google Cloud Console](https://console.cloud.google.com/iam-admin/serviceaccounts) and store it under an `env` folder.
- `GITHUB_TOKEN`: A GitHub personal access token with rate limiting. You can create one on [GitHub Developer Settings](https://github.com/settings/tokens?type=beta).
