CONSTRAINTS = [
    "CREATE CONSTRAINT user_unique "
    "IF NOT EXISTS "
    "FOR (u:User) REQUIRE u.username IS UNIQUE",

    "CREATE CONSTRAINT subreddit_unique "
    "IF NOT EXISTS "
    "FOR (s:Subreddit) REQUIRE s.name IS UNIQUE",

    "CREATE CONSTRAINT post_unique "
    "IF NOT EXISTS "
    "FOR (p:Post) REQUIRE p.post_id IS UNIQUE",

    "CREATE CONSTRAINT comment_unique "
    "IF NOT EXISTS "
    "FOR (c:Comment) "
    "REQUIRE c.comment_id IS UNIQUE",

    "CREATE CONSTRAINT topic_unique "
    "IF NOT EXISTS "
    "FOR (t:Topic) REQUIRE t.name IS UNIQUE",

    "CREATE CONSTRAINT company_unique "
    "IF NOT EXISTS "
    "FOR (co:Company) REQUIRE co.name IS UNIQUE",

    "CREATE CONSTRAINT model_unique "
    "IF NOT EXISTS "
    "FOR (m:Model) REQUIRE m.name IS UNIQUE",

    "CREATE CONSTRAINT period_unique "
    "IF NOT EXISTS "
    "FOR (tp:TimePeriod) "
    "REQUIRE tp.period IS UNIQUE",
]

INDEXES = [
    "CREATE INDEX post_created "
    "IF NOT EXISTS "
    "FOR (p:Post) ON (p.created_at)",

    "CREATE INDEX comment_created "
    "IF NOT EXISTS "
    "FOR (c:Comment) ON (c.created_at)",

    "CREATE INDEX post_sentiment "
    "IF NOT EXISTS "
    "FOR (p:Post) ON (p.sentiment)",

    "CREATE INDEX comment_sentiment "
    "IF NOT EXISTS "
    "FOR (c:Comment) ON (c.sentiment)",
]
