import requests
import json
import csv
from datetime import datetime, timedelta, timezone
import os
import re
import time
from textblob import TextBlob
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pandas as pd
import numpy as np

class TwitterScraper:
    def __init__(self, bearer_token):
        """Initialize the scraper with Twitter API bearer token"""
        self.bearer_token = bearer_token
        self.base_url = "https://api.twitter.com/2"
        self.headers = {
            "Authorization": f"Bearer {bearer_token}",
            "Content-Type": "application/json"
        }
        
        # Initialize NLP components
        self._initialize_nlp()
        
        # Spam detection patterns
        self.spam_patterns = [
            r'(?i)\b(?:click here|free money|make money fast|limited time|act now)\b',
            r'(?i)\b(?:viagra|casino|lottery|winner|congratulations)\b',
            r'(?i)\b(?:buy now|discount|sale|offer|deal)\b',
            r'https?://[^\s]+(?:\.tk|\.ml|\.ga|\.cf)',  # Suspicious domains
            r'(?:https?://[^\s]+){3,}',  # Multiple links
            r'\b(?:[A-Z]{2,}\s*){5,}\b',  # Excessive caps
            r'(?:[@#]\w+\s*){5,}',  # Excessive hashtags/mentions
        ]
    
    def _initialize_nlp(self):
        """Initialize NLTK components"""
        try:
            nltk.download('punkt', quiet=True)
            nltk.download('stopwords', quiet=True)
            nltk.download('wordnet', quiet=True)
            nltk.download('vader_lexicon', quiet=True)
            
            self.stop_words = set(stopwords.words('english'))
            self.lemmatizer = WordNetLemmatizer()
            print("NLP components initialized successfully!")
        except Exception as e:
            print(f"Warning: Could not initialize NLP components: {e}")
            self.stop_words = set()
            self.lemmatizer = None
    
    def _get_time_params(self, time_period="24h", custom_start=None, custom_end=None):
        """
        Generate time parameters for tweet search with API validation
        
        Args:
            time_period (str): Predefined time period ('1h', '24h', '7d', '30d', 'custom')
            custom_start (str): Custom start time in ISO format (YYYY-MM-DDTHH:MM:SSZ)
            custom_end (str): Custom end time in ISO format (YYYY-MM-DDTHH:MM:SSZ)
        
        Returns:
            dict: Time parameters for API request
        """
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Twitter API limitation: can only search tweets from the last 7 days for recent search
        api_limit = now - timedelta(days=7)
        
        # Ensure end_time is at least 30 seconds before now (Twitter API requirement is 10 seconds minimum)
        safe_end_time = now - timedelta(seconds=30)
        
        if time_period == "custom" and custom_start and custom_end:
            # Use custom time range but validate against API limits
            try:
                start_time_dt = datetime.fromisoformat(custom_start.replace('Z', '+00:00')).replace(tzinfo=None)
                end_time_dt = datetime.fromisoformat(custom_end.replace('Z', '+00:00')).replace(tzinfo=None)
                
                # Check if dates are within API limits
                if start_time_dt < api_limit:
                    print(f"⚠️  WARNING: Start date is older than Twitter API limit (7 days)")
                    print(f"   Adjusting start date from {start_time_dt.strftime('%Y-%m-%d %H:%M:%S')} to {api_limit.strftime('%Y-%m-%d %H:%M:%S')}")
                    start_time_dt = api_limit
                
                if end_time_dt < api_limit:
                    print(f"⚠️  WARNING: End date is older than Twitter API limit (7 days)")
                    print(f"   Adjusting end date from {end_time_dt.strftime('%Y-%m-%d %H:%M:%S')} to {safe_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    end_time_dt = safe_end_time
                elif end_time_dt > safe_end_time:
                    print(f"⚠️  WARNING: End date too recent, adjusting to ensure API compliance")
                    print(f"   Adjusting end date from {end_time_dt.strftime('%Y-%m-%d %H:%M:%S')} to {safe_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
                    end_time_dt = safe_end_time
                
                start_time = start_time_dt.isoformat() + "Z"
                end_time = end_time_dt.isoformat() + "Z"
                
            except Exception as e:
                print(f"❌ Error parsing custom dates: {e}")
                print("Using last 24 hours as fallback.")
                start_time = (safe_end_time - timedelta(hours=24)).isoformat() + "Z"
                end_time = safe_end_time.isoformat() + "Z"
        else:
            # Use predefined time periods with API validation
            time_deltas = {
                "1h": timedelta(hours=1),
                "6h": timedelta(hours=6),
                "24h": timedelta(hours=24),
                "3d": timedelta(days=3),
                "7d": timedelta(days=7),
                "30d": timedelta(days=7)  # Limit 30d to 7d due to API restrictions
            }
            
            if time_period not in time_deltas:
                time_period = "24h"  # Default fallback
            
            if time_period == "30d":
                print("⚠️  NOTE: 30-day period limited to 7 days due to Twitter API restrictions")
            
            start_time_dt = safe_end_time - time_deltas[time_period]
            
            # Ensure we don't go beyond API limits
            if start_time_dt < api_limit:
                start_time_dt = api_limit
                print(f"⚠️  Adjusted search period due to Twitter API 7-day limit")
            
            start_time = start_time_dt.isoformat() + "Z"
            end_time = safe_end_time.isoformat() + "Z"
        
        return {
            "start_time": start_time,
            "end_time": end_time
        }
    
    def _enhance_query_for_relevance(self, query):
        """
        Enhance the search query to get more relevant and unique results
        
        Args:
            query (str): Original search query
            
        Returns:
            str: Enhanced query with filters
        """
        # Remove retweets and quoted tweets to get original content
        enhanced_query = f"({query}) -is:retweet -is:quote"
        
        # Filter out replies to focus on original tweets
        enhanced_query += " -is:reply"
        
        # Filter out potential spam/promotional content
        enhanced_query += " -\"buy now\" -\"click here\" -\"free money\" -\"limited time\""
        
        return enhanced_query
    
    def _format_time_display(self, time_str):
        """Format time string for display"""
        try:
            dt = datetime.fromisoformat(time_str.replace('Z', '+00:00'))
            return dt.strftime("%Y-%m-%d %H:%M:%S UTC")
        except:
            return time_str
    
    def search_tweets(self, query, max_results=100, time_period="24h", custom_start=None, custom_end=None, retry_without_time=True):
        """
        Search for tweets based on a query with time filtering and rate limit handling
        
        Args:
            query (str): Search query (topic, hashtag, or keywords)
            max_results (int): Number of tweets to fetch (10-100)
            time_period (str): Time period ('1h', '6h', '24h', '7d', '30d', 'custom', 'none')
            custom_start (str): Custom start time (ISO format) if time_period='custom'
            custom_end (str): Custom end time (ISO format) if time_period='custom'
            retry_without_time (bool): If True, retry without time constraints on rate limit
        
        Returns:
            list: List of extracted tweet data
        """
        # Twitter API v2 endpoint for recent tweet search
        endpoint = f"{self.base_url}/tweets/search/recent"
        
        # Enhanced query for better relevance and filtering out retweets/replies
        enhanced_query = self._enhance_query_for_relevance(query)
        
        # Base parameters without time constraints
        base_params = {
            "query": enhanced_query,
            "max_results": min(max_results, 100),  # API limit is 100
            "tweet.fields": "created_at,author_id,public_metrics,lang,conversation_id,referenced_tweets",
            "expansions": "author_id,referenced_tweets.id",
            "user.fields": "username,name,verified,profile_image_url,description,public_metrics"
        }
        
        # Add time parameters if not 'none'
        if time_period != "none":
            time_params = self._get_time_params(time_period, custom_start, custom_end)
            base_params.update(time_params)
            print(f"Time range: {self._format_time_display(time_params['start_time'])} to {self._format_time_display(time_params['end_time'])}")
        else:
            print("Searching all recent tweets (no time limit)")
        
        print(f"Searching for tweets about: '{query}'...")
        print(f"Enhanced query: '{enhanced_query}'")
        
        try:
            response = requests.get(endpoint, headers=self.headers, params=base_params)
            response.raise_for_status()
            data = response.json()
            
            if "data" not in data or len(data["data"]) == 0:
                print("No tweets found for this query and time range.")
                print("Try:")
                print("   • Using a broader search query")
                print("   • Selecting a different time period")
                print("   • Checking if there's recent activity for this topic")
                return []
            
            # Extract and process tweet data with deduplication and relevance scoring
            tweets = self._extract_tweet_data(data, query)
            
            if tweets:
                # Sort by relevance score (highest first)
                tweets = sorted(tweets, key=lambda x: x.get('relevance_score', 0), reverse=True)
                print(f"Successfully fetched {len(tweets)} unique, relevant tweets!")
            else:
                print("No relevant tweets found after filtering.")
            
            return tweets
            
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error: {e}")
            
            # Handle rate limiting (429 error)
            if response.status_code == 429:
                print("\nRATE LIMIT EXCEEDED")
                print("   Twitter API rate limit: 300 requests per 15 minutes")
                
                if retry_without_time and time_period != "none":
                    print("\n� Retrying without time constraints...")
                    time.sleep(2)  # Brief pause
                    return self.search_tweets(query, max_results, "none", None, None, False)
                else:
                    print("\n💡 Solutions:")
                    print("   • Wait 15 minutes before trying again")
                    print("   • Use a more specific search query to get fewer results")
                    print("   • Try searching without time constraints")
                    
                    # Ask user if they want to wait
                    try:
                        wait_choice = input("\nWait 15 minutes? (y/n, default n): ").strip().lower()
                        if wait_choice == 'y':
                            print("⏳ Waiting 15 minutes for rate limit reset...")
                            time.sleep(900)  # 15 minutes
                            return self.search_tweets(query, max_results, time_period, custom_start, custom_end, False)
                    except KeyboardInterrupt:
                        print("\n❌ Search cancelled.")
                        
                    return []
            
            # Try to parse other error responses
            try:
                error_data = response.json()
                if "errors" in error_data:
                    print("\n🔍 Error Details:")
                    for error in error_data["errors"]:
                        print(f"   • {error.get('message', 'Unknown error')}")
                        
                        # Specific handling for time-related errors
                        if "start_time" in error.get("message", "") or "end_time" in error.get("message", ""):
                            print("\n💡 Twitter API Time Limits:")
                            print("   • Recent search API only covers the last 7 days")
                            print("   • For older tweets, you need Twitter Academic Research access")
                            if retry_without_time:
                                print("   • Retrying without time constraints...")
                                return self.search_tweets(query, max_results, "none", None, None, False)
                            
                        # Specific handling for query errors
                        elif "query" in error.get("message", "").lower():
                            print("\n💡 Query Suggestions:")
                            print("   • Use simpler keywords")
                            print("   • Remove special characters")
                            print("   • Try hashtags like #topic")
                            
            except:
                print(f"Response: {response.text}")
                
            return []
        except Exception as e:
            print(f"❌ Error: {e}")
            print("\n💡 Troubleshooting:")
            print("   • Check your internet connection")
            print("   • Verify your Twitter API bearer token")
            print("   • Ensure your API access level supports recent search")
            return []
    
    def _is_spam(self, tweet_text, author_data):
        """
        Detect if a tweet is likely spam based on content and author metrics
        
        Args:
            tweet_text (str): The tweet content
            author_data (dict): Author information
        
        Returns:
            bool: True if likely spam, False otherwise
        """
        spam_score = 0
        
        # Check for spam patterns in text
        for pattern in self.spam_patterns:
            if re.search(pattern, tweet_text):
                spam_score += 2
        
        # Check for suspicious characteristics
        if len(tweet_text) < 10:  # Too short
            spam_score += 1
        
        if tweet_text.count('#') > 5:  # Excessive hashtags
            spam_score += 2
        
        if tweet_text.count('@') > 3:  # Excessive mentions
            spam_score += 1
        
        # Check author characteristics
        if not author_data.get('verified', False):
            if author_data.get('name', '').isdigit():  # Numeric username
                spam_score += 2
            
            # Check for suspicious profile description
            description = author_data.get('description', '').lower()
            if any(word in description for word in ['follow back', 'dm for', 'earn money', 'free']):
                spam_score += 1
        
        # Return True if spam score is high
        return spam_score >= 3
    
    def _process_with_nlp(self, text):
        """
        Process tweet text with NLP to extract insights
        
        Args:
            text (str): Tweet text
        
        Returns:
            dict: NLP analysis results
        """
        try:
            # Create TextBlob object
            blob = TextBlob(text)
            
            # Sentiment analysis
            sentiment = blob.sentiment
            sentiment_label = 'positive' if sentiment.polarity > 0.1 else 'negative' if sentiment.polarity < -0.1 else 'neutral'
            
            # Clean text for processing
            clean_text = re.sub(r'http\S+|@\w+|#\w+', '', text)  # Remove URLs, mentions, hashtags
            clean_text = re.sub(r'[^a-zA-Z\s]', '', clean_text)  # Remove special characters
            
            # Tokenization and preprocessing
            tokens = word_tokenize(clean_text.lower())
            filtered_tokens = [word for word in tokens if word not in self.stop_words and len(word) > 2]
            
            # Lemmatization
            if self.lemmatizer:
                lemmatized_tokens = [self.lemmatizer.lemmatize(word) for word in filtered_tokens]
            else:
                lemmatized_tokens = filtered_tokens
            
            # Extract keywords (most frequent meaningful words)
            from collections import Counter
            word_freq = Counter(lemmatized_tokens)
            keywords = [word for word, freq in word_freq.most_common(5)]
            
            # Extract hashtags and mentions
            hashtags = re.findall(r'#(\w+)', text)
            mentions = re.findall(r'@(\w+)', text)
            
            return {
                'sentiment_score': round(sentiment.polarity, 3),
                'sentiment_label': sentiment_label,
                'subjectivity': round(sentiment.subjectivity, 3),
                'keywords': keywords,
                'hashtags': hashtags,
                'mentions': mentions,
                'word_count': len(tokens),
                'clean_text': ' '.join(lemmatized_tokens)
            }
            
        except Exception as e:
            print(f"⚠️  NLP processing error: {e}")
            return {
                'sentiment_score': 0.0,
                'sentiment_label': 'neutral',
                'subjectivity': 0.0,
                'keywords': [],
                'hashtags': [],
                'mentions': [],
                'word_count': 0,
                'clean_text': ''
            }
    
    def _extract_tweet_data(self, data, original_query):
        """Extract relevant information from API response with spam filtering, NLP, deduplication and relevance scoring"""
        tweets = data.get("data", [])
        users = {user["id"]: user for user in data.get("includes", {}).get("users", [])}
        
        extracted_data = []
        spam_count = 0
        duplicate_count = 0
        seen_texts = set()  # For deduplication
        seen_users_content = {}  # Track content per user to avoid user spam
        
        for tweet in tweets:
            author_id = tweet.get("author_id")
            author = users.get(author_id, {})
            tweet_text = tweet.get("text", "")
            
            # Skip if this is a retweet or quote tweet (additional safety check)
            referenced_tweets = tweet.get("referenced_tweets", [])
            if any(ref.get("type") in ["retweeted", "quoted"] for ref in referenced_tweets):
                continue
            
            # Check for spam
            if self._is_spam(tweet_text, author):
                spam_count += 1
                print(f"🚫 Filtered spam tweet from @{author.get('username', 'unknown')}")
                continue
            
            # Deduplication: Check for similar content
            tweet_text_normalized = self._normalize_text_for_comparison(tweet_text)
            if self._is_duplicate_content(tweet_text_normalized, seen_texts):
                duplicate_count += 1
                print(f"🔄 Filtered duplicate content from @{author.get('username', 'unknown')}")
                continue
            
            # Check for user content spam (same user posting very similar content)
            username = author.get('username', 'unknown')
            if self._is_user_content_spam(tweet_text_normalized, username, seen_users_content):
                print(f"🔁 Filtered repetitive content from @{username}")
                continue
            
            # Add to seen content
            seen_texts.add(tweet_text_normalized)
            if username not in seen_users_content:
                seen_users_content[username] = []
            seen_users_content[username].append(tweet_text_normalized)
            
            # Process with NLP
            nlp_results = self._process_with_nlp(tweet_text)
            
            # Calculate relevance score
            relevance_score = self._calculate_relevance_score(tweet_text, author, tweet, original_query, nlp_results)
            
            tweet_data = {
                "tweet_id": tweet.get("id"),
                "text": tweet_text,
                "created_at": tweet.get("created_at"),
                "author_id": author_id,
                "author_username": author.get("username", "Unknown"),
                "author_name": author.get("name", "Unknown"),
                "author_verified": author.get("verified", False),
                "author_description": author.get("description", ""),
                "author_followers": author.get("public_metrics", {}).get("followers_count", 0),
                "author_following": author.get("public_metrics", {}).get("following_count", 0),
                "likes": tweet.get("public_metrics", {}).get("like_count", 0),
                "retweets": tweet.get("public_metrics", {}).get("retweet_count", 0),
                "replies": tweet.get("public_metrics", {}).get("reply_count", 0),
                "quotes": tweet.get("public_metrics", {}).get("quote_count", 0),
                "language": tweet.get("lang", "unknown"),
                "conversation_id": tweet.get("conversation_id"),
                "tweet_url": f"https://twitter.com/{author.get('username', 'i')}/status/{tweet.get('id')}",
                
                # NLP results
                "sentiment_score": nlp_results["sentiment_score"],
                "sentiment_label": nlp_results["sentiment_label"],
                "subjectivity": nlp_results["subjectivity"],
                "keywords": ", ".join(nlp_results["keywords"]),
                "hashtags": ", ".join(nlp_results["hashtags"]),
                "mentions": ", ".join(nlp_results["mentions"]),
                "word_count": nlp_results["word_count"],
                "clean_text": nlp_results["clean_text"],
                
                # Relevance scoring
                "relevance_score": relevance_score
            }
            
            extracted_data.append(tweet_data)
        
        # Print filtering statistics
        if spam_count > 0:
            print(f"🛡️  Filtered out {spam_count} spam tweets")
        if duplicate_count > 0:
            print(f"🔄 Filtered out {duplicate_count} duplicate tweets")
        
        return extracted_data
    
    def _normalize_text_for_comparison(self, text):
        """Normalize text for duplicate detection"""
        # Remove URLs, mentions, hashtags, and extra whitespace
        normalized = re.sub(r'http\S+|@\w+|#\w+', '', text)
        normalized = re.sub(r'[^\w\s]', '', normalized)  # Remove punctuation
        normalized = re.sub(r'\s+', ' ', normalized)  # Normalize whitespace
        return normalized.lower().strip()
    
    def _is_duplicate_content(self, text, seen_texts, similarity_threshold=0.8):
        """Check if content is duplicate or very similar to existing content"""
        from difflib import SequenceMatcher
        
        for seen_text in seen_texts:
            similarity = SequenceMatcher(None, text, seen_text).ratio()
            if similarity > similarity_threshold:
                return True
        return False
    
    def _is_user_content_spam(self, text, username, seen_users_content, max_similar_per_user=2):
        """Check if user is posting too much similar content"""
        if username not in seen_users_content:
            return False
        
        user_texts = seen_users_content[username]
        similar_count = 0
        
        for user_text in user_texts:
            from difflib import SequenceMatcher
            similarity = SequenceMatcher(None, text, user_text).ratio()
            if similarity > 0.7:  # Lower threshold for same user
                similar_count += 1
                if similar_count >= max_similar_per_user:
                    return True
        
        return False
    
    def _calculate_relevance_score(self, text, author, tweet, original_query, nlp_results):
        """Calculate relevance score for ranking tweets"""
        score = 0
        
        # Query keyword relevance (case insensitive)
        query_words = original_query.lower().split()
        text_lower = text.lower()
        keyword_matches = sum(1 for word in query_words if word in text_lower)
        score += keyword_matches * 10
        
        # Engagement metrics (normalize to prevent huge tweets from dominating)
        likes = tweet.get("public_metrics", {}).get("like_count", 0)
        retweets = tweet.get("public_metrics", {}).get("retweet_count", 0)
        replies = tweet.get("public_metrics", {}).get("reply_count", 0)
        
        engagement_score = min((likes * 0.5 + retweets * 1.5 + replies * 1.0), 50)
        score += engagement_score
        
        # Author credibility
        if author.get("verified", False):
            score += 15
        
        # Author follower count (normalized)
        followers = author.get("public_metrics", {}).get("followers_count", 0)
        if followers > 1000:
            score += min(followers / 1000, 20)  # Cap at 20 points
        
        # Content quality indicators
        word_count = nlp_results.get("word_count", 0)
        if 10 <= word_count <= 200:  # Prefer meaningful but not too long tweets
            score += 5
        
        # Hashtag relevance
        hashtags = nlp_results.get("hashtags", [])
        relevant_hashtags = sum(1 for hashtag in hashtags if any(word in hashtag.lower() for word in query_words))
        score += relevant_hashtags * 3
        
        # Sentiment bonus for neutral/informative content
        sentiment_label = nlp_results.get("sentiment_label", "neutral")
        if sentiment_label == "neutral":
            score += 3  # Slight preference for neutral/informative content
        
        # Penalize overly subjective content
        subjectivity = nlp_results.get("subjectivity", 0)
        if subjectivity > 0.8:
            score -= 5
        
        # Language preference (English content gets slight boost)
        if tweet.get("lang") == "en":
            score += 2
        
        return round(score, 2)
    
    def save_to_json(self, data, filename=None):
        """Save tweet data to JSON file"""
        if not data:
            print("⚠️  No data to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twitter_comments_{timestamp}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Data saved to {filename}")
    
    def save_to_csv(self, data, filename=None):
        """Save tweet data to CSV file"""
        if not data:
            print("⚠️  No data to save.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twitter_comments_{timestamp}.csv"
        
        # Get all unique keys from all dictionaries
        fieldnames = list(data[0].keys())
        
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data)
        
        print(f"💾 Data saved to {filename}")
    
    def generate_analytics_report(self, data, filename=None):
        """Generate comprehensive analytics report with NLP insights"""
        if not data:
            print("⚠️  No data to analyze.")
            return
        
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"twitter_analytics_report_{timestamp}.txt"
        
        # Calculate analytics
        total_tweets = len(data)
        sentiment_counts = {'positive': 0, 'negative': 0, 'neutral': 0}
        total_engagement = 0
        verified_count = 0
        languages = {}
        all_keywords = []
        all_hashtags = []
        relevance_scores = []
        
        for tweet in data:
            # Sentiment analysis
            sentiment_counts[tweet['sentiment_label']] += 1
            
            # Engagement metrics
            engagement = tweet['likes'] + tweet['retweets'] + tweet['replies']
            total_engagement += engagement
            
            # Verified accounts
            if tweet['author_verified']:
                verified_count += 1
            
            # Languages
            lang = tweet['language']
            languages[lang] = languages.get(lang, 0) + 1
            
            # Keywords and hashtags
            if tweet['keywords']:
                all_keywords.extend(tweet['keywords'].split(', '))
            if tweet['hashtags']:
                all_hashtags.extend(tweet['hashtags'].split(', '))
            
            # Relevance scores
            if 'relevance_score' in tweet:
                relevance_scores.append(tweet['relevance_score'])
        
        # Top keywords and hashtags
        from collections import Counter
        top_keywords = Counter([k for k in all_keywords if k]).most_common(10)
        top_hashtags = Counter([h for h in all_hashtags if h]).most_common(10)
        
        # Generate report
        report = f"""
TWITTER ANALYTICS REPORT
Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
{'='*80}

OVERVIEW
--------
Total Tweets Analyzed: {total_tweets}
Total Engagement: {total_engagement:,} (likes + retweets + replies)
Average Engagement per Tweet: {total_engagement/total_tweets:.1f}
Verified Accounts: {verified_count} ({verified_count/total_tweets*100:.1f}%)

SENTIMENT ANALYSIS
------------------
Positive: {sentiment_counts['positive']} ({sentiment_counts['positive']/total_tweets*100:.1f}%)
Negative: {sentiment_counts['negative']} ({sentiment_counts['negative']/total_tweets*100:.1f}%)
Neutral: {sentiment_counts['neutral']} ({sentiment_counts['neutral']/total_tweets*100:.1f}%)

LANGUAGE DISTRIBUTION
--------------------
"""
        
        for lang, count in sorted(languages.items(), key=lambda x: x[1], reverse=True)[:5]:
            report += f"{lang.upper()}: {count} ({count/total_tweets*100:.1f}%)\n"
        
        report += f"""
TOP KEYWORDS
------------
"""
        for i, (keyword, count) in enumerate(top_keywords, 1):
            report += f"{i:2d}. {keyword} ({count} mentions)\n"
        
        report += f"""
TOP HASHTAGS
------------
"""
        for i, (hashtag, count) in enumerate(top_hashtags, 1):
            report += f"{i:2d}. #{hashtag} ({count} mentions)\n"
        
        report += f"""
ENGAGEMENT INSIGHTS
------------------
Most Liked Tweet: {max(data, key=lambda x: x['likes'])['likes']:,} likes
Most Retweeted: {max(data, key=lambda x: x['retweets'])['retweets']:,} retweets
Most Replies: {max(data, key=lambda x: x['replies'])['replies']:,} replies

CONTENT QUALITY
---------------
Average Word Count: {sum(tweet['word_count'] for tweet in data)/total_tweets:.1f} words
Average Subjectivity Score: {sum(tweet['subjectivity'] for tweet in data)/total_tweets:.3f}
(0 = objective, 1 = subjective)

RELEVANCE SCORING
-----------------"""
        
        if relevance_scores:
            avg_relevance = sum(relevance_scores) / len(relevance_scores)
            max_relevance = max(relevance_scores)
            min_relevance = min(relevance_scores)
            report += f"""
Average Relevance Score: {avg_relevance:.2f}
Highest Relevance Score: {max_relevance:.2f}
Lowest Relevance Score: {min_relevance:.2f}
Note: Higher scores indicate more relevant and engaging content
"""
        else:
            report += "\nRelevance scoring data not available\n"
        
        report += f"""
{'='*80}
"""
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📊 Analytics report saved to {filename}")
        print("\n" + "="*50)
        print("QUICK INSIGHTS")
        print("="*50)
        print(f"📈 {sentiment_counts['positive']} positive, {sentiment_counts['negative']} negative tweets")
        print(f"🔥 Top keyword: {top_keywords[0][0] if top_keywords else 'N/A'}")
        print(f"# Top hashtag: #{top_hashtags[0][0] if top_hashtags else 'N/A'}")
        print(f"💬 Avg engagement: {total_engagement/total_tweets:.1f} per tweet")
    
    def display_preview(self, data, num_tweets=5):
        """Display a preview of the fetched tweets with NLP insights"""
        if not data:
            return
        
        print("\n" + "="*80)
        print(f"PREVIEW OF FETCHED TWEETS (showing {min(num_tweets, len(data))} of {len(data)})")
        print("="*80 + "\n")
        
        for i, tweet in enumerate(data[:num_tweets], 1):
            print(f"Tweet #{i} (Relevance Score: {tweet.get('relevance_score', 'N/A')})")
            print(f"Author: {tweet['author_name']} (@{tweet['author_username']})")
            if tweet['author_verified']:
                print("✓ Verified Account")
            followers = tweet.get('author_followers', 0)
            if followers > 0:
                followers_str = f"{followers:,}" if followers < 1000 else f"{followers/1000:.1f}K"
                print(f"👥 Followers: {followers_str}")
            print(f"Posted: {tweet['created_at']}")
            print(f"Text: {tweet['text'][:200]}{'...' if len(tweet['text']) > 200 else ''}")
            print(f"Engagement: ❤️ {tweet['likes']} | 🔁 {tweet['retweets']} | 💬 {tweet['replies']}")
            
            # NLP insights
            sentiment_emoji = "😊" if tweet['sentiment_label'] == 'positive' else "😞" if tweet['sentiment_label'] == 'negative' else "😐"
            print(f"Sentiment: {sentiment_emoji} {tweet['sentiment_label'].upper()} (score: {tweet['sentiment_score']})")
            
            if tweet['keywords']:
                print(f"Keywords: {tweet['keywords']}")
            if tweet['hashtags']:
                print(f"Hashtags: #{tweet['hashtags'].replace(', ', ' #')}")
            
            print(f"URL: {tweet['tweet_url']}")
            print("-" * 80 + "\n")
    
    def get_time_selection(self):
        """Interactive time period selection with FROM/TO dates"""
        print("\n" + "="*60)
        print("📅 DATE RANGE SELECTION")
        print("="*60)
        print("Choose your preferred date filtering method:")
        print("1. Quick time periods (last 1h, 24h, 7d, or no time limit)")
        print("2. Custom date range (FROM date TO date)")
        
        try:
            method_choice = input("\nSelect method (1-2, default 1): ").strip() or "1"
            
            if method_choice == "2":
                return self._get_custom_date_range()
            else:
                return self._get_quick_time_period()
                
        except KeyboardInterrupt:
            print("\n❌ Selection cancelled. Using last 24 hours as default.")
            return "24h", None, None
    
    def _get_quick_time_period(self):
        """Get quick time period selection"""
        print("\n🕐 QUICK TIME PERIODS")
        print("⚠️  Note: Twitter API only allows searching tweets from the last 7 days")
        print("")
        print("Choose a time period:")
        print("1. Last 1 hour")
        print("2. Last 6 hours")
        print("3. Last 24 hours (default)")
        print("4. Last 3 days")
        print("5. Last 7 days (maximum)")
        print("6. Last 30 days (limited to 7 days)")
        print("7. No time limit (search all recent tweets)")
        
        try:
            choice = input("\nEnter your choice (1-7, default 3): ").strip() or "3"
            
            time_periods = {
                "1": "1h",
                "2": "6h", 
                "3": "24h",
                "4": "3d",
                "5": "7d",
                "6": "30d",
                "7": "none"
            }
            
            time_period = time_periods.get(choice, "24h")
            
            if choice == "6":
                print("📅 Selected time period: 30d (limited to 7d due to API restrictions)")
            elif choice == "7":
                print("📅 Selected: No time limit (searching all recent tweets)")
            else:
                print(f"📅 Selected time period: {time_period}")
            
            return time_period, None, None
            
        except KeyboardInterrupt:
            print("\n❌ Selection cancelled. Using 24h as default.")
            return "24h", None, None
    
    def _get_custom_date_range(self):
        """Get custom FROM and TO date range"""
        print("\n📅 CUSTOM DATE RANGE")
        print("⚠️  IMPORTANT: Twitter API only allows searching tweets from the last 7 days")
        print("   Dates older than 7 days will be automatically adjusted")
        print("")
        print("Enter your preferred date range:")
        print("Format options:")
        print("  • YYYY-MM-DD (e.g., 2024-11-01)")
        print("  • YYYY-MM-DD HH:MM (e.g., 2024-11-01 14:30)")
        print("  • Relative: 'yesterday', 'last week', '3 days ago'")
        
        try:
            # Get FROM date
            print("\n📅 FROM DATE:")
            from_input = input("Enter start date/time: ").strip()
            
            # Get TO date
            print("\n📅 TO DATE:")
            to_input = input("Enter end date/time (leave blank for 'now'): ").strip()
            
            if not from_input:
                print("❌ FROM date is required. Using last 24 hours as default.")
                return "24h", None, None
            
            # Parse dates
            from_date = self._parse_date_input(from_input)
            to_date = self._parse_date_input(to_input) if to_input else datetime.now(timezone.utc).replace(tzinfo=None)
            
            if from_date and to_date:
                # Check API limitations
                now = datetime.now(timezone.utc).replace(tzinfo=None)
                api_limit = now - timedelta(days=7)
                
                # Warn about API limitations
                if from_date < api_limit or to_date < api_limit:
                    print(f"\n⚠️  WARNING: Selected dates are older than Twitter API limit")
                    print(f"   API limit: {api_limit.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                    print(f"   Dates will be automatically adjusted to fit API constraints")
                
                # Ensure from_date is before to_date
                if from_date > to_date:
                    from_date, to_date = to_date, from_date
                    print("🔄 Swapped dates to ensure FROM is before TO")
                
                # Convert to ISO format
                from_iso = from_date.isoformat() + "Z"
                to_iso = to_date.isoformat() + "Z"
                
                print(f"📅 Selected range:")
                print(f"   FROM: {from_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                print(f"   TO:   {to_date.strftime('%Y-%m-%d %H:%M:%S UTC')}")
                
                # Calculate duration for display
                duration = to_date - from_date
                if duration.days > 0:
                    duration_str = f"{duration.days} days, {duration.seconds//3600} hours"
                else:
                    duration_str = f"{duration.seconds//3600} hours, {(duration.seconds%3600)//60} minutes"
                
                print(f"   DURATION: {duration_str}")
                
                return "custom", from_iso, to_iso
            else:
                print("❌ Invalid date format. Using last 24 hours as default.")
                return "24h", None, None
                
        except KeyboardInterrupt:
            print("\n❌ Selection cancelled. Using last 24 hours as default.")
            return "24h", None, None
    
    def _parse_date_input(self, date_input):
        """Parse various date input formats"""
        date_input = date_input.lower().strip()
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        
        # Handle relative dates
        if date_input == "yesterday":
            return now - timedelta(days=1)
        elif date_input == "last week":
            return now - timedelta(weeks=1)
        elif "days ago" in date_input:
            try:
                days = int(date_input.split()[0])
                return now - timedelta(days=days)
            except:
                pass
        elif "hours ago" in date_input:
            try:
                hours = int(date_input.split()[0])
                return now - timedelta(hours=hours)
            except:
                pass
        elif "minutes ago" in date_input:
            try:
                minutes = int(date_input.split()[0])
                return now - timedelta(minutes=minutes)
            except:
                pass
        
        # Handle absolute dates
        date_formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%d %H:%M",
            "%Y-%m-%d",
            "%m/%d/%Y %H:%M",
            "%m/%d/%Y",
            "%d-%m-%Y %H:%M",
            "%d-%m-%Y",
            "%Y/%m/%d %H:%M",
            "%Y/%m/%d"
        ]
        
        for fmt in date_formats:
            try:
                return datetime.strptime(date_input, fmt)
            except ValueError:
                continue
        
        print(f"⚠️  Could not parse date: '{date_input}'")
        return None


def main():
    """Main function to run the scraper with advanced date filtering"""
    print("="*80)
    print("TWITTER COMMENT SCRAPER WITH ADVANCED DATE FILTERING")
    print("="*80 + "\n")
    
    # Display API limitations info
    print("ℹ️  TWITTER API INFORMATION")
    print("-" * 40)
    print("• Recent Search API covers the last 7 days only")
    print("• For older tweets, Academic Research access is required")
    print("• Rate limits: 300 requests per 15-minute window")
    print("• Each request can fetch up to 100 tweets maximum")
    print("")
    
    # Get bearer token from user
    bearer_token = input("Enter your Twitter API Bearer Token: ").strip()
    
    if not bearer_token:
        print("❌ Bearer token is required!")
        print("\n💡 How to get a Twitter API Bearer Token:")
        print("   1. Go to https://developer.twitter.com/")
        print("   2. Create a Twitter Developer account")
        print("   3. Create a new app")
        print("   4. Copy the Bearer Token from your app settings")
        return
    
    # Get search query
    query = input("Enter topic/keyword to search (e.g., 'python programming', '#AI'): ").strip()
    
    if not query:
        print("❌ Search query is required!")
        return
    
    # Get number of results
    try:
        max_results = int(input("Enter max results (10-100, default 50): ").strip() or "50")
        max_results = max(10, min(100, max_results))
    except ValueError:
        max_results = 50
    
    # Initialize scraper
    scraper = TwitterScraper(bearer_token)
    
    # Get time selection
    time_period, custom_start, custom_end = scraper.get_time_selection()
    
    # Fetch tweets with time filtering
    tweets = scraper.search_tweets(query, max_results, time_period, custom_start, custom_end)
    
    if tweets:
        # Display preview
        scraper.display_preview(tweets)
        
        # Save to files with time info in filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        time_suffix = f"_{time_period}" if time_period != "custom" else "_custom"
        
        json_filename = f"twitter_comments_{timestamp}{time_suffix}.json"
        csv_filename = f"twitter_comments_{timestamp}{time_suffix}.csv"
        report_filename = f"twitter_analytics_report_{timestamp}{time_suffix}.txt"
        
        scraper.save_to_json(tweets, json_filename)
        scraper.save_to_csv(tweets, csv_filename)
        scraper.generate_analytics_report(tweets, report_filename)
        
        print("\n✨ Scraping completed successfully!")
        print(f"📊 Generated {len(tweets)} clean tweets with full NLP analysis!")
        print(f"⏰ Time period: {time_period}")
    else:
        print("\n❌ No tweets were fetched.")
        print("\n💡 Common solutions:")
        print("   • Try a different search query or hashtag")
        print("   • Use a recent time period (last 24h, 7d)")
        print("   • Check if there's recent activity for this topic")
        print("   • Verify your API credentials and access level")


if __name__ == "__main__":
    main()