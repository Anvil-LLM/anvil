#!/bin/bash

SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
INSTALLER="${REPO_ROOT}/docs/anvil-nvidia-install.sh"
INSTALL_SH="${REPO_ROOT}/docs/install.sh"

PASS=0
FAIL=0

report() {
    if [ "$1" = pass ]; then
        echo "  [PASS] $2"; PASS=$((PASS + 1))
    else
        echo "  [FAIL] $2"; FAIL=$((FAIL + 1))
    fi
}

ROOT=""
fake_env() {
    ROOT=$(mktemp -d)
    mkdir -p "${ROOT}/bin"

    cat > "${ROOT}/os-release" <<'EOF'
ID=ubuntu
VERSION_ID="24.04"
PRETTY_NAME="Ubuntu 24.04 LTS"
EOF
    printf 'nouveau 12345 0 - Live 0xffffffff\n' > "${ROOT}/modules"

    cat > "${ROOT}/bin/uname" <<'EOF'

case "$1" in
    -s) echo Linux ;;
    -m) echo x86_64 ;;
    -r) echo 6.8.0-45-generic ;;
    *) /usr/bin/uname "$@" ;;
esac
EOF

    cat > "${ROOT}/bin/curl" <<'EOF'

out=""; url=""; prev=""
for a in "$@"; do
    [ "$a" = "-o" ] && out=""
    [ -n "$prev" ] && [ "$prev" = "-o" ] && out=$a
    prev=$a; url=$a
done
if [ -n "$out" ]; then

    printf '#!/bin/sh\ncase "$1" in --version) echo "anvil v9.9.9 (fake)"; exit 0 ;; *) exit 0 ;; esac\n' > "$out"
    exit 0
fi
case "$url" in
    *latest.txt) echo "595.84 595.84/NVIDIA-Linux-x86_64-595.84.run" ;;
    */) printf '%s\n' '580.173.02/' '580.178.04/' '470.239.06/' '470.256.02/' '595.84/' ;;
    *) echo 'not-found'; exit 1 ;;
esac
EOF

    printf '#!/bin/sh\nif [ "$1" = "-nn" ]; then echo "%s [10de:1234]"; else echo "%s"; fi\n' "$1" "$1" > "${ROOT}/bin/lspci"
    chmod +x "${ROOT}/bin/"*
}

fake_installer() {
    sed -e "s|/etc/os-release|${ROOT}/os-release|" -e "s|/proc/modules|${ROOT}/modules|" "$INSTALLER" > "${ROOT}/installer.sh"
    chmod +x "${ROOT}/installer.sh"
}
run_nv() {
    PATH="${ROOT}/bin:$PATH" sh "${ROOT}/installer.sh" "$@" 2>&1
}

test_branches() {
    echo "── Group: branch selection (universal mode, --dry-run)"

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    fake_installer
    OUT=$(run_nv --dry-run)
    echo "$OUT" | grep -q '595.84' && report pass "RTX 4090 -> latest 595.84" || report fail "RTX 4090 branch"
    echo "$OUT" | grep -q -- '--no-nouveau-check' && report pass "nouveau loaded -> --no-nouveau-check flag" || report fail "nouveau flag"

    fake_env 'VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
    fake_installer
    OUT=$(run_nv --dry-run)
    echo "$OUT" | grep -q '580.178.04' && report pass "GTX 1080 (Pascal) -> 580 branch" || report fail "Pascal branch (got: $(echo "$OUT" | grep 'Driver' || true))"

    fake_env 'VGA compatible controller: NVIDIA Corporation GK110 [GeForce GTX 780]'
    fake_installer
    OUT=$(run_nv --dry-run)
    echo "$OUT" | grep -q '470.256.02' && report pass "GTX 780 (Kepler) -> 470 branch" || report fail "Kepler branch"

    fake_env '3D controller: NVIDIA Corporation TU104GL [Tesla T4]'
    fake_installer
    OUT=$(run_nv --dry-run)
    echo "$OUT" | grep -q '595.84' && report pass "Tesla T4 (Turing) -> latest 595.84" || report fail "Tesla T4 branch"

    fake_env 'VGA compatible controller: NVIDIA Corporation VGA compatible controller'
    fake_installer
    OUT=$(run_nv --dry-run)
    echo "$OUT" | grep -q '595.84' && report pass "unknown generation -> latest fallback" || report fail "unknown generation fallback"

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    fake_installer
    OUT=$(run_nv --dry-run --driver 580.178.04)
    echo "$OUT" | grep -q '580.178.04' && report pass "--driver pinning overrides branch" || report fail "--driver pinning"
}

test_modes() {
    echo "── Group: modes, flags, guards"

    fake_env 'VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
    fake_installer
    OUT=$(run_nv --check)
    echo "$OUT" | grep -q 'GPU generation: pascal' && report pass "--check classifies Pascal" || report fail "--check classification"
    echo "$OUT" | grep -q '580 series' && report pass "--check shows 580 branch note" || report fail "--check branch note"

    fake_env 'VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
    fake_installer
    OUT=$(run_nv --distro --dry-run)
    echo "$OUT" | grep -q 'Mode        : distro' && report pass "--distro mode selected" || report fail "--distro mode"
    echo "$OUT" | grep -q 'apt-get install' && report pass "--distro prints apt commands" || report fail "--distro apt commands"

    fake_env 'VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
    fake_installer
    ERR=$(run_nv </dev/null; echo "exit=$?")
    echo "$ERR" | grep -q 'non-interactive' && report pass "non-tty without --yes refuses (no auto-sudo)" || report fail "non-tty guard (got: $ERR)"

    fake_env 'VGA compatible controller: NVIDIA Corporation GP104 [GeForce GTX 1080]'
    fake_installer
    OUT=$(run_nv --help)
    echo "$OUT" | grep -qi 'universal NVIDIA driver installer' && report pass "--help renders" || report fail "--help"
}

test_install_sh() {
    echo "── Group: install.sh NVIDIA hook"

    run_inst() {
        mkdir -p "${ROOT}/home"
        PATH="${ROOT}/bin:$PATH" HOME="${ROOT}/home" INSTALL_DIR="${ROOT}/tgt" ANVIL_VERSION=v9.9.9 ANVIL_SKIP_CHECKSUM=1 \
            sh "$INSTALL_SH" "$@" 2>&1
    }

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    OUT=$(run_inst)
    echo "$OUT" | grep -q 'anvil-linux-x86_64-cuda' && report pass "NVIDIA GPU -> prefers CUDA asset" || report fail "CUDA asset selection"
    echo "$OUT" | grep -q 'driver is not installed' && report pass "no driver -> installer offer shown" || report fail "driver offer"

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    OUT=$(run_inst)
    echo "$OUT" | grep -q 'curl -fsSL -O' && report pass "offer uses download-then-sudo (no auto-sudo)" || report fail "offer safety"

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    cat > "${ROOT}/bin/curl" <<'EOF'

out=""; url=""; prev=""
for a in "$@"; do
    [ "$a" = "-o" ] && out=""
    [ -n "$prev" ] && [ "$prev" = "-o" ] && out=$a
    prev=$a; url=$a
done
if [ -n "$out" ]; then
    printf '%s' "$url" | grep -q -- '-cuda' && exit 1

    printf '#!/bin/sh\ncase "$1" in --version) echo "anvil v9.9.9 (fake)"; exit 0 ;; *) exit 0 ;; esac\n' > "$out"
    exit 0
fi
echo '{"tag_name":"v9.9.9"}'
EOF
    chmod +x "${ROOT}/bin/curl"
    OUT=$(run_inst)
    echo "$OUT" | grep -q 'trying fallback' && report pass "CUDA download fail -> falls back to plain asset" || report fail "CUDA fallback"

    fake_env 'VGA compatible controller: NVIDIA Corporation AD102 [GeForce RTX 4090]'
    printf '#!/bin/sh\necho 595.84\n' > "${ROOT}/bin/nvidia-smi"
    chmod +x "${ROOT}/bin/nvidia-smi"
    OUT=$(run_inst)
    echo "$OUT" | grep -q 'driver is not installed' && report fail "offer shown despite driver present" || report pass "driver present -> no offer"

    fake_env 'VGA compatible controller: Advanced Micro Devices [AMD/ATI] Navi 21 [Radeon RX 6800 XT]'
    OUT=$(run_inst --nvidia)
    echo "$OUT" | grep -q 'forcing the CUDA build' && report pass "--nvidia forces CUDA asset" || report fail "--nvidia force"

    fake_env 'VGA compatible controller: Advanced Micro Devices [AMD/ATI] Navi 21 [Radeon RX 6800 XT]'
    OUT=$(run_inst)
    echo "$OUT" | grep -q -- '--cuda' && report fail "CUDA asset used without --nvidia on AMD" || report pass "no GPU/no flag -> plain asset"
}

run_all() {
    echo "== anvil installer smoke test =="
    echo "== installer: ${INSTALLER}"
    echo "== install.sh: ${INSTALL_SH}"
    echo ""
    test_branches
    echo ""
    test_modes
    echo ""
    test_install_sh
}

if [ -n "$1" ]; then
    case "$1" in
        branches) test_branches ;;
        modes)    test_modes ;;
        installsh) test_install_sh ;;
        *) echo "unknown group: $1"; exit 2 ;;
    esac
else
    run_all
fi

echo ""
echo "══════════════════════════════════════"
echo "  TOTAL: ${PASS} pass, ${FAIL} fail"
echo "══════════════════════════════════════"
[ "$FAIL" -eq 0 ]
