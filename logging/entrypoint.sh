#!/bin/bash
# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.



# Check if the fluentbit-key secret exists in the default namespace
if kubectl get secret fluentbit-key &>/dev/null; then
    echo "Secret fluentbit-key found. Starting Fluent Bit..."
    exec /opt/fluent-bit/bin/fluent-bit -c /fluent-bit/etc/fluent-bit.conf & 
    FLUENT_BIT_PID=$!
    echo "Fluent Bit started with PID $FLUENT_BIT_PID"
else
    echo "Secret fluentbit-key not found. Waiting gracefully for app to terminate."
fi

# Check if the python3 process is still running
while kill -0 "$(pgrep -f 'python3 .* tee /var/log/.*\.log')" 2>/dev/null; do
    sleep 60;
done

if [[ -n "$FLUENT_BIT_PID" ]]; then  # Check if FLUENT_BIT_PID is set and not empty
    if kill -TERM "$FLUENT_BIT_PID" 2>/dev/null; then
        # Give Fluent Bit some time to shut down gracefully
        echo "Sending SIGTERM to Fluent Bit with PID $FLUENT_BIT_PID"
        sleep 5
        kill -9 "$FLUENT_BIT_PID" 2>/dev/null # Force kill if it doesn't stop
    fi
fi

echo "Logging container terminating."
exit 0
