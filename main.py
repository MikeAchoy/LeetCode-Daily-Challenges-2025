
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
January 4, 2025
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
January 5, 2025
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
January 6, 2025
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
1400. Construct K Palindrome Strings
Given a string s and an integer k, return true if you can use all the characters in s to construct k palindrome strings or false otherwise.
'''
def canConstruct(s: str, k: int) -> bool:
    # Handle edge cases
    if len(s) < k:
        return False

    if len(s) == k:
        return True

    # Initialize frequency dictionary and odd_count
    freq = [0] * 26
    odd_count = 0

    # Increment the value of the index corresponding to the current character
    for char in s:
        freq[ord(char) - ord("a")] += 1

    # Count the number of characters that appear an odd number of times in s
    for count in freq:
        if count % 2 == 1:
            odd_count += 1

    # Return if the number of odd frequencies is less than or equal to k
    return odd_count <= k


'''
January 12, 2025
2116. Check if a Parentheses String Can Be Valid:
A parentheses string is a non-empty string consisting only of '(' and ')'. It is valid if any of the following conditions is true:
It is ().
It can be written as AB (A concatenated with B), where A and B are valid parentheses strings.
It can be written as (A), where A is a valid parentheses string.
You are given a parentheses string s and a string locked, both of length n. locked is a binary string consisting only of '0's and '1's. For each index i of locked,
If locked[i] is '1', you cannot change s[i].
But if locked[i] is '0', you can change s[i] to either '(' or ')'.
Return true if you can make s a valid parentheses string. Otherwise, return false.
'''
def canBeValid(s: str, locked: str) -> bool:
    length = len(s)

    # If length of string is odd, return false.
    if length % 2 == 1:
        return False

    open_brackets = []
    unlocked = []

    # Iterate through the string to handle '(' and ')'
    for i in range(length):
        if locked[i] == "0":
            unlocked.append(i)
        elif s[i] == "(":
            open_brackets.append(i)
        elif s[i] == ")":
            if open_brackets:
                open_brackets.pop()
            elif unlocked:
                unlocked.pop()
            else:
                return False

    # Match remaining open brackets and the unlocked characters
    while open_brackets and unlocked and open_brackets[-1] < unlocked[-1]:
        open_brackets.pop()
        unlocked.pop()

    if open_brackets:
        return False

    return True


'''
January 13, 2025
3223. Minimum Length of String After Operations:
You are given a string s.
You can perform the following process on s any number of times:
Choose an index i in the string such that there is at least one character to the left of index i that is equal to s[i], and at least one character to the right that is also equal to s[i].
Delete the closest character to the left of index i that is equal to s[i].
Delete the closest character to the right of index i that is equal to s[i].
Return the minimum length of the final string s that you can achieve.
'''
from collections import Counter

def minimumLength(s: str) -> int:
    # Step 1: Count the frequency of each character in the string
    char_frequency_map = Counter(s)

    # Step 2: Calculate the number of characters to delete
    delete_count = 0
    for frequency in char_frequency_map.values():
        if frequency % 2 == 1:
            # If frequency is odd, delete all except one
            delete_count += frequency - 1
        else:
            # If frequency is even, delete all except two
            delete_count += frequency - 2

    # Step 3: Return the minimum length after deletions
    return len(s) - delete_count


'''
January 14, 2025
2657. Find the Prefix Common Array of Two Array:
You are given two 0-indexed integer permutations A and B of length n.
A prefix common array of A and B is an array C such that C[i] is equal to the count of numbers that are present at or before the index i in both A and B.
Return the prefix common array of A and B.
A sequence of n integers is called a permutation if it contains all integers from 1 to n exactly once.
'''
def findThePrefixCommonArray(A: list[int], B: list[int]) -> list[int]:
    n = len(A)
    prefix_common_array = [0] * n

    # Loop through each index to calculate common elements for each prefix
    for current_index in range(n):
        common_count = 0

        # Compare elements in A and B within the range of current prefix
        for a_index in range(current_index + 1):
            for b_index in range(current_index + 1):

                # Check if elements match, and count if they do
                if A[a_index] == B[b_index]:
                    common_count += 1
                    break  # Prevent counting duplicates

        # Store the count of common elements for the current prefix
        prefix_common_array[current_index] = common_count

    # Return the final list with counts of common elements in each prefix
    return prefix_common_array


'''
January 15, 2025
2429. Minimize XOR:
Given two positive integers num1 and num2, find the positive integer x such that:
x has the same number of set bits as num2, and
The value x XOR num1 is minimal.
Note that XOR is the bitwise XOR operation.
Return the integer x. The test cases are generated such that x is uniquely determined.
The number of set bits of an integer is the number of 1's in its binary representation.
'''
def minimizeXor(num1: int, num2: int) -> int:
    # Helper functions for setting and checking bits:
    # Helper function to check if the given bit position in x is set (1)
    def _is_set(x: int, bit: int) -> bool:
        return (x & (1 << bit)) != 0

    # Helper function to set the given bit position in x to 1
    def _set_bit(x: int, bit: int):
        return x | (1 << bit)

    result = 0

    target_set_bits_count = bin(num2).count("1")
    set_bits_count = 0
    current_bit = 31  # Start from the most significant bit

    # While result has fewer set bits than num2
    while set_bits_count < target_set_bits_count:
        # If the current bit of num1 is set or we must set all remaining bits in result
        if _is_set(num1, current_bit) or (
            target_set_bits_count - set_bits_count > current_bit
        ):
            result = _set_bit(result, current_bit)
            set_bits_count += 1

        current_bit -= 1  # Move to the next bit
    return result


'''
January 16, 2025
2425. Bitwise XOR of All Pairings
You are given two 0-indexed arrays, nums1 and nums2, consisting of non-negative integers. There exists another array, nums3, which contains the bitwise XOR of all pairings of integers between nums1 and nums2 (every integer in nums1 is paired with every integer in nums2 exactly once).
Return the bitwise XOR of all integers in nums3.
'''
def xorAllNums(nums1: list[int], nums2: list[int]) -> int:
    # Get lengths of arrays
    len1, len2 = len(nums1), len(nums2)

    # Dictionary to store frequency of each number
    freq = {}

    # Add frequencies for nums1 elements
    # Each element appears n2 times in final result
    for num in nums1:
        freq[num] = freq.get(num, 0) + len2

    # Add frequencies for nums2 elements
    # Each element appears n1 times in final result
    for num in nums2:
        freq[num] = freq.get(num, 0) + len1

    # XOR numbers that appear odd number of times
    ans = 0
    for num in freq:
        if freq[num] % 2:
                ans ^= num
    return ans


'''
January 17, 2025
2683. Neighboring Bitwise XORa:
A 0-indexed array derived with length n is derived by computing the bitwise XOR (⊕) of adjacent values in a binary array original of length n.
Specifically, for each index i in the range [0, n - 1]:
If i = n - 1, then derived[i] = original[i] ⊕ original[0].
Otherwise, derived[i] = original[i] ⊕ original[i + 1].
Given an array derived, your task is to determine whether there exists a valid binary array original that could have formed derived.
Return true if such an array exists or false otherwise.
A binary array is an array containing only 0's and 1's
'''
def doesValidArrayExist(derived: list[int]) -> bool:
    # Create an original array initialized with 0.
    original = [0]
    for i in range(len(derived)):
        original.append(derived[i] ^ original[i])

    # Store the validation results in checkForZero and checkForOne respectively.
    check_for_zero = original[0] == original[-1]
    original = [1]
    for i in range(len(derived)):
        original.append(derived[i] ^ original[i])
    check_for_one = original[0] == original[-1]

    return check_for_zero or check_for_one


'''
January 18, 2025
1368. Minimum Cost to Make at Least One Valid Path in a Grid:
Given an m x n grid. Each cell of the grid has a sign pointing to the next cell you should visit if you are currently in this cell. The sign of grid[i][j] can be:

1 which means go to the cell to the right. (i.e go from grid[i][j] to grid[i][j + 1])
2 which means go to the cell to the left. (i.e go from grid[i][j] to grid[i][j - 1])
3 which means go to the lower cell. (i.e go from grid[i][j] to grid[i + 1][j])
4 which means go to the upper cell. (i.e go from grid[i][j] to grid[i - 1][j])
Notice that there could be some signs on the cells of the grid that point outside the grid.

You will initially start at the upper left cell (0, 0). A valid path in the grid is a path that starts from the upper left cell (0, 0) and ends at the bottom-right cell (m - 1, n - 1) following the signs on the grid. The valid path does not have to be the shortest.

You can modify the sign on a cell with cost = 1. You can modify the sign on a cell one time only.

Return the minimum cost to make the grid have at least one valid path.
'''
import collections
# Direction vectors: right, left, down, up (matching grid values 1,2,3,4)
_dirs = [(0, 1), (0, -1), (1, 0), (-1, 0)]

# Check if coordinates are within grid bounds
def _is_valid(
    row: int, col: int, num_rows: int, num_cols: int
) -> bool:
    return 0 <= row < num_rows and 0 <= col < num_cols

def minCost(grid: list[list[int]]) -> int:
    num_rows, num_cols = len(grid), len(grid[0])

    # Track minimum cost to reach each cell
    min_cost = [[float("inf")] * num_cols for _ in range(num_rows)]
    min_cost[0][0] = 0

    # Use deque for 0-1 BFS - add zero cost moves to front, cost=1 to back
    deque = collections.deque([(0, 0)])

    while deque:
        row, col = deque.popleft()

        # Try all four directions
        for dir_idx, (dx, dy) in enumerate(_dirs):
            new_row, new_col = row + dx, col + dy
            cost = 0 if grid[row][col] == dir_idx + 1 else 1

                # If position is valid and we found a better path
            if (
                _is_valid(new_row, new_col, num_rows, num_cols)
                and min_cost[row][col] + cost < min_cost[new_row][new_col]
            ):

                min_cost[new_row][new_col] = min_cost[row][col] + cost

                # Add to back if cost=1, front if cost=0
                if cost == 1:
                    deque.append((new_row, new_col))
                else:
                    deque.appendleft((new_row, new_col))

    return int(min_cost[num_rows - 1][num_cols - 1])


def main():
    pass


if __name__ == '__main__':
    main()

