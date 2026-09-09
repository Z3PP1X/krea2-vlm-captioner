"""Compliance and copyright safety utilities.

Implements:
1. robots.txt parsing and caching per domain.
2. Text and Data Mining (TDM) opt-out signal detection under § 44b UrhG (EU Copyright Directive Art. 4).
   - Inspects HTTP headers (e.g. 'tdm-reservation: 1')
   - Inspects HTML meta tags (<meta name="tdm-reservation" content="1">, <meta name="robots" content="noai, noimageai">)
"""

from __future__ import annotations

import logging
import urllib.robotparser
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse, urljoin
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("pipeline.crawler.compliance")


class ComplianceChecker:
    """Manages robots.txt caching and TDM reservation checking per domain."""

    def __init__(self, user_agent: str, respect_robots_txt: bool = True, filter_tdm: bool = True):
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.filter_tdm = filter_tdm
        self._robots_cache: Dict[str, urllib.robotparser.RobotFileParser] = {}

    def get_domain(self, url: str) -> str:
        """Extracts scheme + netloc from a URL."""
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"

    def is_url_allowed_by_robots(self, url: str, session: Optional[requests.Session] = None) -> bool:
        """Checks if URL path is allowed by robots.txt."""
        if not self.respect_robots_txt:
            return True

        domain = self.get_domain(url)
        if domain not in self._robots_cache:
            rp = urllib.robotparser.RobotFileParser()
            robots_url = urljoin(domain, "/robots.txt")
            rp.set_url(robots_url)
            try:
                if session:
                    resp = session.get(robots_url, timeout=10)
                    if resp.status_code == 200:
                        rp.parse(resp.text.splitlines())
                    else:
                        rp.allow_all = True
                else:
                    rp.read()
            except Exception as exc:
                logger.debug(f"Could not read robots.txt for {domain}: {exc}")
                rp.allow_all = True
            self._robots_cache[domain] = rp

        parser = self._robots_cache[domain]
        return parser.can_fetch(self.user_agent, url)

    def check_tdm_reservation(
        self,
        headers: Optional[Dict[str, str]] = None,
        html_content: Optional[str] = None,
    ) -> Tuple[bool, Optional[str]]:
        """Checks if a page or resource explicitly declares a TDM reservation under § 44b UrhG.
        
        Returns:
            (is_reserved: bool, reason: Optional[str])
        """
        if not self.filter_tdm:
            return False, None

        # 1. Check HTTP Headers (W3C TDM Reservation Protocol & custom headers)
        if headers:
            for k, v in headers.items():
                k_lower = k.lower()
                v_lower = v.lower()
                if k_lower == "tdm-reservation" and v_lower in ["1", "true", "yes"]:
                    return True, "HTTP header 'tdm-reservation: 1' (§ 44b UrhG)"
                if k_lower == "x-robots-tag" and any(token in v_lower for token in ["noai", "noimageai"]):
                    return True, f"X-Robots-Tag '{v}' opt-out"

        # 2. Check HTML Meta Tags
        if html_content:
            try:
                soup = BeautifulSoup(html_content, "html.parser")
                # <meta name="tdm-reservation" content="1">
                tdm_meta = soup.find("meta", attrs={"name": lambda x: x and x.lower() == "tdm-reservation"})
                if tdm_meta and tdm_meta.get("content", "").strip() in ["1", "true", "yes"]:
                    return True, "HTML meta 'tdm-reservation=1' (§ 44b UrhG)"

                # <meta name="robots" content="noai, noimageai">
                robots_meta = soup.find_all("meta", attrs={"name": lambda x: x and x.lower() in ["robots", "googlebot"]})
                for m in robots_meta:
                    content = m.get("content", "").lower()
                    if "noai" in content or "noimageai" in content:
                        return True, f"HTML meta robots '{content}' AI opt-out"
            except Exception as exc:
                logger.debug(f"HTML TDM meta parse error: {exc}")

        return False, None
