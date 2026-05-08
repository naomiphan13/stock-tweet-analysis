-- SQLite
-- Exploratory
SELECT *
FROM stocks;

SELECT *
FROM tweet_stocks;

SELECT *
FROM tweets;

SELECT *
FROM users
ORDER BY users.name;

-- Show the list of tweets by stock symbol
SELECT stock_symbol as symbol, text
FROM tweets JOIN tweet_stocks ON (tweets.id = tweet_stocks.tweet_id)
ORDER BY created_at DESC;

-- The number of tweet mentions per stock
SELECT stock_symbol, COUNT(tweet_id) AS num_tweets
FROM tweet_stocks
GROUP BY stock_symbol;

-- The number of tweets per stock symbol and per user (matrix)
SELECT
    stock_symbol AS company,
    SUM(CASE WHEN users.name = 'Cassandra Unchained' THEN 1 ELSE 0 END) AS "Cassandra Unchained",
    SUM(CASE WHEN users.name = 'Cathie Wood' THEN 1 ELSE 0 END) AS "Cathie Wood",
    SUM(CASE WHEN users.name = 'Donald J. Trump' THEN 1 ELSE 0 END) AS "Donald J. Trump",
    SUM(CASE WHEN users.name = 'Elon Musk' THEN 1 ELSE 0 END) AS "Elon Musk",
    SUM(CASE WHEN users.name = 'Jim Cramer' THEN 1 ELSE 0 END) AS "Jim Cramer",
    SUM(CASE WHEN users.name = 'Ray Dalio' THEN 1 ELSE 0 END) AS "Ray Dalio"
FROM tweet_stocks
    JOIN tweets ON tweet_stocks.tweet_id = tweets.id 
    JOIN users ON  users.id = tweets.user_id
GROUP BY stock_symbol
ORDER BY stock_symbol;

-- Find the number of tweets created per day per company
SELECT
    DATE(created_at) as Date,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'NVDA' THEN 1 ELSE 0 END) AS NVDA,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'AAPL' THEN 1 ELSE 0 END) AS AAPL,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'MSFT' THEN 1 ELSE 0 END) AS MSFT,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'GOOG' THEN 1 ELSE 0 END) AS GOOG,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'AMZN' THEN 1 ELSE 0 END) AS AMZN,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'AVGO' THEN 1 ELSE 0 END) AS AVGO,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'META' THEN 1 ELSE 0 END) AS META,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'TSM' THEN 1 ELSE 0 END) AS TSM,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'TSLA' THEN 1 ELSE 0 END) AS TSLA,
    SUM(CASE WHEN tweet_stocks.stock_symbol = 'TCEHY' THEN 1 ELSE 0 END) AS TCEHY
FROM tweets JOIN tweet_stocks ON tweets.id = tweet_stocks.tweet_id
GROUP BY Date
ORDER BY Date;

-- Extract tweets text for sentiment analysis
SELECT
    stock_symbol as Stock, 
    tweets.text as Tweet,
    DATE(created_at) as Date
FROM tweets JOIN tweet_stocks ON tweets.id = tweet_stocks.tweet_id
ORDER BY Date;






