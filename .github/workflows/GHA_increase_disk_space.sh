#!/bin/bash
df -h
# locations from the internet, e.g. https://github.com/easimon/maximize-build-space
# saves about 2GB
if [ -d /usr/share/dotnet ]; then
    echo removing dotnet
    sudo rm -rf /usr/share/dotnet
fi
# saves about 10 GB
if [ -d "$AGENT_TOOLSDIRECTORY" ]; then
    echo removing agent_tools
    sudo rm -rf "$AGENT_TOOLSDIRECTORY"
fi
# saves about 10 GB
if [ -d /usr/local/lib/android ]; then
    echo removing android files
    sudo rm -rf /usr/local/lib/android
fi
if [ -d /opt/ghc ]; then
    echo removing android files
    sudo rm -rf /opt/ghc
fi
df -h
