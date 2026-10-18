# Wi-Fi Connected but No Internet

Applies to: Windows and macOS laptops on office or home networks.

## Quick checks
Confirm other devices on the same Wi-Fi can reach the internet. If they cannot, the problem is the router or ISP, not the laptop. Restart the router and wait two minutes.

## Check the IP address
Open a terminal and run `ipconfig` (Windows) or `ifconfig` (macOS). If the address starts with 169.254.x.x the laptop failed to get an address from DHCP. Run `ipconfig /release` then `ipconfig /renew`, or toggle Wi-Fi off and on.

## DNS failures
If you can ping 8.8.8.8 but websites do not load, DNS is failing. Flush the cache with `ipconfig /flushdns` (Windows) or `sudo dscacheutil -flushcache` (macOS). Then set DNS manually to 8.8.8.8 and 1.1.1.1 in adapter settings.

## Captive portals and proxies
Hotel or guest Wi-Fi may require a login page. Open http://neverssl.com to trigger it. Also check that no stale proxy is set under Settings > Network > Proxy.

## Reset the network stack
As a last resort on Windows run `netsh winsock reset` and `netsh int ip reset` in an admin prompt, then reboot.
