import re

# =============================================
# EXTRACTION PATTERNS
# =============================================

URL_PATTERN = re.compile(
    r"https?://[^\s\)\]\>\"\',]+"
)

# Strict: must have r/ NOT preceded by / or word char
REDDIT_USER_PATTERN = re.compile(
    r"(?<![/\w])u/(\w+)"
)

# Strict: must have r/ NOT preceded by / or word char,
# name starts with letter, 2-21 chars
SUBREDDIT_PATTERN = re.compile(
    r"(?<![/\w])r/([A-Za-z]\w{1,20})"
)

MARKDOWN_LINK_PATTERN = re.compile(
    r"\[([^\]]*)\]\(([^\)]+)\)"
)

# Handles both full words and Reddit shorthand:
# "3 hours ago", "3hr ago", "3h ago",
# "1mo ago", "2d ago", "1y ago", "5m ago"
RELATIVE_TIME_PATTERN = re.compile(
    r"(\d+)\s*"
    r"(seconds?|secs?|"
    r"minutes?|mins?|"
    r"hours?|hrs?|"
    r"days?|"
    r"weeks?|wks?|"
    r"months?|mos?|"
    r"years?|yrs?|"
    r"h|d|w|m|y)"
    r"\.?\s*ago",
    flags=re.IGNORECASE,
)

ABSOLUTE_DATE_PATTERN = re.compile(
    r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)"
    r"\s+(\d{1,2}),?\s+(\d{4})",
    flags=re.IGNORECASE,
)

# =============================================
# AI ENTITY PATTERNS
# =============================================

AI_COMPANY_PATTERN = re.compile(
    r"\b(OpenAI|Anthropic|Google|DeepMind|"
    r"Meta|Microsoft|Mistral|Cohere|"
    r"Stability\s?AI|Hugging\s?Face|xAI|"
    r"Perplexity|Inflection|Nvidia)\b",
    flags=re.IGNORECASE,
)

AI_MODEL_PATTERN = re.compile(
    r"\b(GPT[-\s]?4o?|GPT[-\s]?3\.5|GPT[-\s]?5|"
    r"Claude\s*(?:Sonnet|Opus|Haiku|Fable)?(?:\s*[\d\.]+)?|"
    r"Gemini(?:\s*(?:Pro|Ultra|Flash|Nano))?(?:\s*[\d\.]+)?|"
    r"Llama[-\s]?[\d\.]+|Llama|"
    r"Mistral(?:\s*(?:Large|Medium|Small))?(?:\s*[\d\.]+)?|"
    r"Command[-\s]?R\+?|"
    r"DALL[-\s]?E[-\s]?[\d]?|"
    r"Stable\s*Diffusion(?:\s*[\d\.]+)?|"
    r"Whisper|Codex|"
    r"Phi[-\s]?[\d\.]+|"
    r"Qwen[-\s]?[\d\.]+|"
    r"DeepSeek(?:\s*(?:Coder|V\d))?|"
    r"Grok(?:\s*[\d\.]+)?|"
    r"Sora|Midjourney)\b",
    flags=re.IGNORECASE,
)

# Canonical names for normalization
MODEL_NORMALIZATION = {
    "gpt-4": "GPT-4", "gpt 4": "GPT-4",
    "gpt4": "GPT-4",
    "gpt-4o": "GPT-4o", "gpt 4o": "GPT-4o",
    "gpt4o": "GPT-4o",
    "gpt-3.5": "GPT-3.5", "gpt 3.5": "GPT-3.5",
    "gpt-5": "GPT-5", "gpt 5": "GPT-5",
    "claude": "Claude",
    "claude sonnet": "Claude Sonnet",
    "claude opus": "Claude Opus",
    "claude haiku": "Claude Haiku",
    "claude fable": "Claude Fable",
    "gemini": "Gemini",
    "gemini pro": "Gemini Pro",
    "gemini ultra": "Gemini Ultra",
    "gemini flash": "Gemini Flash",
    "llama": "Llama",
    "mistral": "Mistral",
    "deepseek": "DeepSeek",
    "grok": "Grok",
    "sora": "Sora",
    "midjourney": "Midjourney",
    "dall-e": "DALL-E", "dall e": "DALL-E",
    "stable diffusion": "Stable Diffusion",
    "whisper": "Whisper",
    "codex": "Codex",
    "command r": "Command R",
    "command-r": "Command R",
}

# =============================================
# REMOVAL PATTERNS — Markdown
# =============================================

HTML_TAG_PATTERN = re.compile(
    r"<[^>]+>"
)

HTML_ENTITY_PATTERN = re.compile(
    r"&(?:amp|lt|gt|nbsp|quot|#\d+|#x[\da-fA-F]+);"
)

MARKDOWN_BOLD_PATTERN = re.compile(
    r"\*\*(.+?)\*\*"
)

MARKDOWN_ITALIC_PATTERN = re.compile(
    r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)"
)

MARKDOWN_STRIKETHROUGH_PATTERN = re.compile(
    r"~~(.+?)~~"
)

MARKDOWN_HEADING_PATTERN = re.compile(
    r"^#{1,6}\s+", flags=re.MULTILINE
)

MARKDOWN_BLOCKQUOTE_PATTERN = re.compile(
    r"^>\s?", flags=re.MULTILINE
)

MARKDOWN_CODE_BLOCK_PATTERN = re.compile(
    r"```[\s\S]*?```"
)

MARKDOWN_INLINE_CODE_PATTERN = re.compile(
    r"`([^`]+)`"
)

MARKDOWN_IMAGE_PATTERN = re.compile(
    r"!\[([^\]]*)\]\([^\)]+\)"
)

MARKDOWN_HR_PATTERN = re.compile(
    r"^[\s]*[-\*_]{3,}[\s]*$",
    flags=re.MULTILINE,
)

# =============================================
# REMOVAL PATTERNS — Reddit noise
# =============================================

DELETED_PATTERN = re.compile(
    r"^\[(?:deleted|removed)\]$",
    flags=re.IGNORECASE | re.MULTILINE,
)

BOT_AUTHOR_PATTERN = re.compile(
    r"^(AutoModerator|RemindMeBot|"
    r"sneakpeekbot|RepostSleuthBot|"
    r"SaveVideo|WikiSummarizerBot|"
    r"haikusbot|MAGIC_EYE_BOT|"
    r"sub_doesnt_exist_bot|"
    r"VisualMod|QualityVote|"
    r"CommonMisspellingBot|"
    r"HelperBot_|GifReversingBot|"
    r"FatFingerHelperBot|"
    r"Anti-Evil Operations)$",
    flags=re.IGNORECASE,
)

AWARD_PATTERN = re.compile(
    r"(?:gold|silver|platinum|helpful|"
    r"wholesome)\s*award",
    flags=re.IGNORECASE,
)

VOTE_NOISE_PATTERN = re.compile(
    r"^\d+\s*(?:upvotes?|downvotes?|points?)"
    r"[\s•]*\d*\s*(?:comments?)?",
    flags=re.IGNORECASE | re.MULTILINE,
)

TIMESTAMP_NOISE_PATTERN = re.compile(
    r"(?:posted|submitted|edited)\s+"
    r"(?:by\s+)?(?:u/\w+\s+)?"
    r"\d+\s+(?:hours?|minutes?|days?|"
    r"weeks?|months?|years?)\s+ago",
    flags=re.IGNORECASE,
)

# Reddit ad detection
AD_PATTERN = re.compile(
    r"(?:^|\n)\s*(?:\[?\w*\]?\s*•?\s*)?"
    r"Promoted\b.*$",
    flags=re.IGNORECASE | re.MULTILINE,
)

AD_DOMAINS = [
    "interserver.net", "lenovo.com",
    "squarespace.com", "grammarly.com",
    "nordvpn.com", "surfshark.com",
    "expressvpn.com", "audible.com",
    "skillshare.com", "brilliant.org",
    "cloudiway.com", "pulumi.com",
    "ibm.com/products",
    "google.com/ads", "ads.google.com",
]

NOISE_KEYWORDS = [
    "Log In", "Sign Up", "Get the Reddit app",
    "Reddit Inc", "User Agreement",
    "Privacy Policy", "Content Policy",
    "Cookie Notice", "Advertise on Reddit",
    "Download the official Reddit app",
    "Open sort options", "Best Comments",
    "Top Comments", "Home", "Popular",
    "Explore", "Terms", "View in App",
    "Continue with Google", "Continue with Email",
    "Continue with Apple", "Create an account",
    "Already a redditor?", "Forgot password",
    "Get the app", "More posts you may like",
    "Read more", "Share", "Report",
    "Collapse video player",
    "People also ask about",
    "Top Posts", "New Comments",
    "View Entire Discussion",
    "Single comment thread",
    "Learn More", "Promoted",
    "About Community", "Community Details",
    "Created ", "Online Members",
    "r/ Rules", "Back to Top",
    "Reddit Premium", "Reddit Coins",
    "More from this community",
    "Reddit Inc ©", "Accessibility",
    "All rights reserved",
    "Community Guidelines",
    "Meet IBM Bob", "IBM Bob",
    "Sponsored", "Ad •",
]

REDDIT_ASSET_PATTERNS = [
    "redditstatic",
    "redditmedia",
    "preview.redd.it",
    "avatars",
    "styles.redditmedia",
    "i.redd.it",
    "v.redd.it",
    "external-preview.redd.it",
]

# =============================================
# WHITESPACE
# =============================================

EXCESS_NEWLINES_PATTERN = re.compile(
    r"\n{3,}"
)

EXCESS_SPACES_PATTERN = re.compile(
    r"[ \t]{2,}"
)

UNICODE_JUNK_PATTERN = re.compile(
    r"[​‌‍﻿ ]"
)

EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F1E0-\U0001F1FF"
    "\U00002700-\U000027BF"
    "\U0000FE00-\U0000FE0F"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "]+",
    flags=re.UNICODE,
)
