
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


'''
Janurary 5, 2025
2381. Shifting Letters II:
You are given a string s of lowercase English letters and a 2D integer array shifts where shifts[i] = [starti, endi, directioni]. For every i, shift the characters in s from the index starti to the index endi (inclusive) forward if directioni = 1, or shift the characters backward if directioni = 0.
Shifting a character forward means replacing it with the next letter in the alphabet (wrapping around so that 'z' becomes 'a'). Similarly, shifting a character backward means replacing it with the previous letter in the alphabet (wrapping around so that 'a' becomes 'z').
Return the final string after all such shifts to s are applied.
'''
def shiftingLetters(s: str, shifts: list[list[int]]) -> str:
    n = len(s)
    diff_array = [0] * n  # Initialize a difference array with all elements set to 0
    
    # Process eahc shift op
    for shift in shifts:
        if shift[2] == 1: # shift dir is forward (1)
            diff_array[shift[0]] += 1 # increment at start index
            if shift[1] + 1 < n:
                diff_array[
                    shift[1] + 1
                ] -= 1 # Decrement at end+1 index
        else: # Else the dir is backward
            diff_array[shift[0]] -= 1 # Decrement at the start index
            if shift[1] + 1 < n:
                diff_array[ 
                    shift[1] + 1
                ] += 1 # Increment at end+1 index

    result = list(s)
    number_of_shifts = 0
    
    # Apply shifts to string
    for i in range(n):
        number_of_shifts = (number_of_shifts + diff_array[i]) % 26
        if number_of_shifts < 0:
            number_of_shifts += 26 # ensure non-neg shifts

        # Calculate the new character by shifting `s[i]`
        shifted_char = chr(
            (ord(s[i]) - ord("a") + number_of_shifts) % 26 + ord("a")
        )
        result[i] = shifted_char

    return "".join(result)


'''
Janurary 6, 2025
1769. Minimun number of Operations to Move All Balls to Each Box:
You have n boxes. You are given a binary string boxes of length n, where boxes[i] is '0' if the ith box is empty, and '1' if it contains one ball.

In one operation, you can move one ball from a box to an adjacent box. Box i is adjacent to box j if abs(i - j) == 1. Note that after doing so, there may be more than one ball in some boxes.

Return an array answer of size n, where answer[i] is the minimum number of operations needed to move all the balls to the ith box.

Each answer[i] is calculated considering the initial state of the boxes.
'''
def minOperations(boxes: str) -> list[int]:
    answer = [0] * len(boxes)
    for current_box in range(len(boxes)):
        # If the current box contains a ball, calculate the number of moves for every box.
        if boxes[current_box] == "1":
            for new_position in range(len(boxes)):
                answer[new_position] += abs(new_position - current_box)
    return answer


'''
January 7. 2025
1408. String Marching in an Array:
Given an array of string words, return all strings in words that is a substring of another word. You can return the answer in any order.
A substring is a contiguous sequence of characters within a string
'''
def stringMatching(words: list[str]) -> list[str]:
    matching_words = []

    # Iterate through each word in the input list.
    for current_word_index in range(len(words)):
        # Compare the current word with all other words.
        for other_word_index in range(len(words)):
            # Skip comparing the word with itself.
            if current_word_index == other_word_index:
                continue
            if words[current_word_index] in words[other_word_index]:
                # Add it to the result list if true.
                matching_words.append(words[current_word_index])
                break  # No need to check further for this word.
    return matching_words


'''
January 8, 2025
3042. Count Prefix and Suffix Pairs I:
You are given a 0-indexed string array words.
Let's define a boolean function isPrefixAndSuffix that takes two strings, str1 and str2:
isPrefixAndSuffix(str1, str2) returns true if str1 is both a 
prefix
 and a 
suffix
 of str2, and false otherwise.
For example, isPrefixAndSuffix("aba", "ababa") is true because "aba" is a prefix of "ababa" and also a suffix, but isPrefixAndSuffix("abc", "abcd") is false.
Return an integer denoting the number of index pairs (i, j) such that i < j, and isPrefixAndSuffix(words[i], words[j]) is true.
'''
def countPrefixSuffixPairs(words: list[str]) -> int:
    n = len(words)
    count = 0

    # Step 1: Iterate through each pair of words
    for i in range(n):
        for j in range(i + 1, n):
            str1 = words[i]
            str2 = words[j]

            # Step 2: Skip if the first str is larger than the second
            if len(str1) > len(str2):
                continue

            # Step 3: Check if str1 is both the prefix and suffix of str2
            if str2.startswith(str1) and str2.endswith(str1):
                count += 1
    return count

'''
January 9, 2025
2185. Counting Words With a Given Prefix
You are given an array of strings words and a string pref.
Return the number of strings in words that contain pref as a prefix.
A prefix of a string s is any leading contiguous substring of s.
'''
def prefixCount(words: list[str], pref: str) -> int:
    count = 0
    # Go through each word
    for word in words:
        # Check if it has the prefix
        if word[:len(pref)] == pref:
            # Increase count if it does
            count += 1
    # One liner solution:
    # return sum(word.startswith(pref) for word in words)
    return count


'''
January 10, 2025
916. Word Subsets:
You are given two string arrays words1 and words2.
A string b is a subset of string a if every letter in b occurs in a including multiplicity.
For example, "wrr" is a subset of "warrior" but is not a subset of "world".
A string a from words1 is universal if for every string b in words2, b is a subset of a.
Return an array of all the universal strings in words1. You may return the answer in any order.
'''
def wordsSubsets(words1: list[str], words2: list[str]) -> list[str]:
    def count(word):
        ans = [0] * 26
        for letter in word:
            ans[ord(letter) - ord('a')] += 1
        return ans

    bmax = [0] * 26
    for b in words2:
        for i, c in enumerate(count(b)):
            bmax[i] = max(bmax[i], c)

    ans = []
    for a in words1:
        if all(x >= y for x, y in zip(count(a), bmax)):
            ans.append(a)
    return ans


'''
January 11, 2025

'''


def main():
    words = ["pay","attention","practice","attend"] 
    pref = "at"
    print(prefixCount(words, pref))
    pass


if __name__ == '__main__':
    main()


