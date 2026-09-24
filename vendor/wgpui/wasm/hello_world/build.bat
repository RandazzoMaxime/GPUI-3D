@echo off
REM Build and serve the WASM hello world example.
REM Uses serve.ps1 for building and serving.

powershell -ExecutionPolicy Bypass -File "%~dp0serve.ps1" %*
