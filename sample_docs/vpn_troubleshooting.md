# VPN Troubleshooting Guide

## VPN will not connect
Verify the internet works without the VPN first. Check the date and time are correct; certificate-based VPNs fail if the clock is off by more than a few minutes. Make sure the gateway address matches the one in the IT onboarding email.

## Error: authentication failed
Reset your directory password and retry. If MFA is enabled, confirm the authenticator app clock is synced. Repeated failures lock the account for 15 minutes.

## Connected but cannot reach internal sites
This is usually split tunneling or DNS. Disconnect and reconnect, then run `nslookup intranet.company.local`. If it fails, set the VPN adapter DNS to the corporate DNS servers. Check that the VPN adapter metric is lower than the Wi-Fi adapter.

## VPN drops every few minutes
Switch from UDP to TCP in the client settings, disable power saving on the Wi-Fi adapter, and update the VPN client to the latest version.
