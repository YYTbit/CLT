from itertools import combinations
from urllib.parse import urlsplit, urlunsplit

import tldextract


# Use the bundled public suffix list; certification never fetches a live list.
_suffixes = tldextract.TLDExtract(suffix_list_urls=(), include_psl_private_domains=True)


def normalize_url(url):
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.hostname or parts.username or parts.password:
        raise ValueError("Source URLs require a public HTTP(S) origin without credentials")
    host = parts.hostname.lower().rstrip(".")
    port = parts.port
    if port and (parts.scheme, port) not in {("http", 80), ("https", 443)}:
        host += f":{port}"
    return urlunsplit((parts.scheme.lower(), host, parts.path.rstrip("/") or "/", parts.query, ""))


def verified_origin(document):
    metadata = document.get("origin", {})
    if metadata.get("status") != "verified":
        return None
    try:
        root = normalize_url(metadata["cited_root"])
        final_url = normalize_url(document.get("redirects", [document["url"]])[-1])
        relation = metadata.get("relation", "primary")
        if relation == "primary":
            if final_url != root:
                return None
        elif relation in {"quotation", "republication"}:
            citations = {normalize_url(url) for url in document.get("cited_urls", [])}
            if root not in citations:
                return None
        else:
            return None
        domain = _suffixes(urlsplit(root).hostname).top_domain_under_public_suffix
    except (KeyError, ValueError, IndexError):
        return None
    return (domain, root) if domain else None


def independent(left, right):
    return left is not None and right is not None and left[0] != right[0] and left[1] != right[1]


def has_independent_support(origins, count):
    if count < 1:
        raise ValueError("Independent-source counts must be positive")
    origins = sorted(set(origin for origin in origins if origin is not None))
    return any(all(independent(left, right) for left, right in combinations(group, 2))
               for group in combinations(origins, count))
