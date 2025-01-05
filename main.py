

'''
First problem of the year, January 1, 2025
1422. Maximum Score After Splitting a String:

Given a string s of zeros and ones, return the maximum score after splitting the string into two non-empty substrings (i.e. left substring and right substring).

The score after splitting a string is the number of zeros in the left substring plus the number of ones in the right substring.
'''


def maxScore(s: str) -> int:
    max_score = 0
    # Iterate through the string, not splitting at last char 
    for i in range(1, len(s)):
        left = s[:i]     # Left part of split
        right = s[i:]    # Right part of split
    
        score = left.count('0') + right.count('1')
        max_score = max(max_score, score)
    return max_score


'''
January 2, 2025
2559. Count Vowel Strings in Ranges:
You are given a 0-indexed array of strings words and a 2D array of integers queries.

Each query queries[i] = [li, ri] asks us to find the number of strings present in the range li to ri (both inclusive) of words that start and end with a vowel.

Return an array ans of size queries.length, where ans[i] is the answer to the ith query.

Note that the vowel letters are 'a', 'e', 'i', 'o', and 'u'.
'''
def vowelStrings(words: list[str], queries: list[list[int]]) -> list[int]:
    vowels = {'a', 'e', 'i', 'o', 'u'}
    n = len(words)
    vowel_count = [0] * n
    prefix_sum = [0] * (n + 1)

    # Preprocess to check if words start and end with vowels
    for i in range(n):
        if words[i][0] in vowels and words[i][-1] in vowels:
            vowel_count[i] = 1

    # Build prefix sum array (need to figure out how these things work)
    for i in range(1, n + 1):
        prefix_sum[i] = prefix_sum[i - 1] + vowel_count[i - 1]

    # Process queries using prefix sum
    query_answers = []
    for q in queries:
        left, right = q[0], q[1]
        count = prefix_sum[right + 1] - prefix_sum[left]
        query_answers.append(count)
    return query_answers


'''
January 3, 2025
2270. Number of Ways to Split Array:
You are given a 0-indexed integer array nums of length n. 
nums contains a valid split at index 1 if the following are true:
    The sum of the first i+1 elements is greater than or qual to the sum of the last n-i-1 elements.
    There is at least one element to the right of i. That is. 0 <= i < n-1
Return the number of valid split in nums. 
'''
def waysToSplitArray(nums: list[int]) -> int:
    # Keep track of sum of elements on left and right sides
    left_sum = 0
    right_sum = 0

    # Initially all elements are on right side
    right_sum = sum(nums)

    # Try each possible split position
    count = 0
    for i in range(len(nums) - 1):
        # Move current element from right to left side
        left_sum += nums[i]
        right_sum -= nums[i]

        # Check if this creates a valid split
        if left_sum >= right_sum:
            count += 1

    return count

'''
Janurary 4, 2025
1930. Unique Length-3 Palindromic Subsequences:
Given a string s, return the number of unique palindromes of length three that are a subsequence of s.
Note that even if there are multiple ways to obtain the same subsequence, it is still only counted once.
A palindrome is a string that reads the same forwards and backwards.
A subsequence of a string is a new string generated from the original string with some characters (can be none) deleted without changing the relative order of the remaining characters.

For example, "ace" is a subsequence of "abcde".
'''
def countPalindromicSubsequence(s: str) -> int:
    letters = set(s)
    count = 0
        
    for letter in letters:
        i = s.index(letter)
        j = s.rindex(letter)

        between = set()
            
        for k in range(i + 1, j):
            between.add(s[k])

        count += len(between)

    return count


def main():
    pass


if __name__ == '__main__':
    main()


