#!/usr/bin/env bash
set -euo pipefail

# Run the module-1 minimal loop: GT labels -> selector windows -> strict audit.

bash refine_v2/commands/00_build_contact_labels.sh
bash refine_v2/commands/01_select_windows.sh
bash refine_v2/commands/02_audit_windows.sh

