# Running the GitHub Actions workflow locally with `act`

The GitHub Actions CI workflow can be tested locally using [`act`](https://github.com/nektos/act).
This is useful for debugging workflow changes before pushing to GitHub.

## Prerequisities

Install `act` and make sure Docker or Podman is availabel and running.

On Linux, the workflow can be run with an Ubuntu 24.04
 container image compatible with GitHub Actions:

```bash
act -W .github/workflows/build-test.yml \
  -P ubuntu-24.04=ghcr.io/catthehacker/ubuntu:act-24.04 \
  --env ACT=true
```

* The ```-W``` option selects the workflow file to run
* The ```-P``` option maps the GitHub Actions
runner label ```ubuntu-24.04``` to a local container image.
The image in the command above is suggested online.
* The ```--env ACT=true``` option sets an environment variable
 used by the workflow to detect that it is running under ```act```.
 Some GitHub setp are skipped.

## NOTES

* It is highly recommended to run only one job at the time.
STIR has an array of different OSes and options.
Don't try to spin them up all together in your local workstation.
* Runnning the workflow locally with ```act``` is not always identical to GitHub.
* If Docker runs out of disk space, remove old images and containers before running

```bash
docker system prune
```
