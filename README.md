# Character Detection

1. Create a directory called `electric` in `azure fileshare`
2. Upload `electric` folder in this repo to the `electric` folder in your fileshare
3. Set environment variables in `env.sh`
4. Create a cluster and jobs with `batchai`

## Set Environment Variables

```sh
export AZURE_BATCHAI_STORAGE_ACCOUNT="..."
export AZURE_FILE_URL="..."
export AZURE_BATCHAI_STORAGE_KEY="..."
export ADMIN_USERNAME="..."
export ADMIN_PASSWORD="..."
export ADMIN_SSH_KEY="..."
```