// Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

// According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes v and w as the lowest node in T that has both v and w as descendants (where we allow a node to be a descendant of itself).”

//         _______3______
//        /              \
//     ___5__          ___1__
//    /      \        /      \
//    6      _2       0       8
//          /  \
//          7   4
// For example, the lowest common ancestor (LCA) of nodes 5 and 1 is 3. Another example is LCA of nodes 5 and 4 is 5, since a node can be a descendant of itself according to the LCA definition.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class LowestCommonAncestorOfABinaryTree {
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if(root == null || root == p || root == q) {
            return root;
        }
        
        TreeNode left = lowestCommonAncestor(root.left, p, q);
        TreeNode right = lowestCommonAncestor(root.right, p, q);
        
        if(left != null && right != null) {
            return root;
        }
        
        return left == null ? right : left;
    }
}
// Invert a binary tree.

//      4
//    /   \
//   2     7
//  / \   / \
// 1   3 6   9

// to

//      4
//    /   \
//   7     2
//  / \   / \
// 9   6 3   1

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class InvertBinaryTree {
    public TreeNode invertTree(TreeNode root) {
        if(root == null) {
            return root;
        }
        
        TreeNode temp = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(temp);
        
        return root;
    }
}
// Find the sum of all left leaves in a given binary tree.

// Example:

//     3
//    / \
//   9  20
//     /  \
//    15   7

// There are two left leaves in the binary tree, with values 9 and 15 respectively. Return 24.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class SumOfLeftLeaves {
    public int sumOfLeftLeaves(TreeNode root) {
        if(root == null) {
            return 0;
        }
        
        int total = 0;
        
        if(root.left != null) {
            if(root.left.left == null && root.left.right == null) {
                total += root.left.val;
            } else {
                total += sumOfLeftLeaves(root.left);
            }
        }
        
        total += sumOfLeftLeaves(root.right);
        
        return total;
    }
}
// Given a binary search tree and a node in it, find the in-order successor of that node in the BST.

// Note: If the given node has no in-order successor in the tree, return null.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class InorderSuccessorInBST {
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        TreeNode successor = null;
        
        while(root != null) {
            if(p.val < root.val) {
                successor = root;
                root = root.left;
            } else {
                root = root.right;
            }
        }
        
        return successor;
    }
}
// Given a binary tree, return all root-to-leaf paths.

// For example, given the following binary tree:

//    1
//  /   \
// 2     3
//  \
//   5
// All root-to-leaf paths are:

// ["1->2->5", "1->3"]

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class BinaryTreePaths {
    public List<String> binaryTreePaths(TreeNode root) {
        List<String> result = new ArrayList<String>();

        if(root == null) {
            return result;
        }
        
        helper(new String(), root, result);
        
        return result;
    }
    
    public void helper(String current, TreeNode root, List<String> result) {
        if(root.left == null && root.right == null) {
            result.add(current + root.val);
        }

        if(root.left != null) {
            helper(current + root.val + "->", root.left, result);
        }

        if(root.right != null) {
            helper(current + root.val + "->", root.right, result);
        }
    }
}
// Given a binary tree, find the maximum path sum.

// For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

// For example:
// Given the below binary tree,

//        1
//       / \
//      2   3
// Return 6.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class BinaryTreeMaximumPathSum {
    int max = Integer.MIN_VALUE;
    
    public int maxPathSum(TreeNode root) {
        maxPathSumRecursive(root);
        return max;
    }
    
    private int maxPathSumRecursive(TreeNode root) {
        if(root == null) {
            return 0;
        }
        
        int left = Math.max(maxPathSumRecursive(root.left), 0);
        int right = Math.max(maxPathSumRecursive(root.right), 0);
        
        max = Math.max(max, root.val + left + right);
        
        return root.val + Math.max(left, right);
    }
}
// Given a binary tree, determine if it is a valid binary search tree (BST).

// Assume a BST is defined as follows:

// The left subtree of a node contains only nodes with keys less than the node's key.
// The right subtree of a node contains only nodes with keys greater than the node's key.
// Both the left and right subtrees must also be binary search trees.
// Example 1:
//     2
//    / \
//   1   3
// Binary tree [2,1,3], return true.
// Example 2:
//     1
//    / \
//   2   3
// Binary tree [1,2,3], return false.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class ValidateBinarySearchTree {
    public boolean isValidBST(TreeNode root) {
        if(root == null) {
            return true;
        }
        
        return validBSTRecursive(root, Long.MIN_VALUE, Long.MAX_VALUE);
    }
    
    public boolean validBSTRecursive(TreeNode root, long minValue, long maxValue) {
        if(root == null) {
            return true;
        } else if(root.val >= maxValue || root.val <= minValue) {
            return false;
        } else {
            return validBSTRecursive(root.left, minValue, root.val) && validBSTRecursive(root.right, root.val, maxValue);
        }
    }
}
//Given a binary search tree and the lowest and highest boundaries as L and R, trim the 
//tree so that all its elements lies in [L, R] (R >= L). You might need to change the root 
//of the tree, so the result should return the new root of the trimmed binary search tree.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
class TrimABinarySearchTree {
    public TreeNode trimBST(TreeNode root, int L, int R) {
        if(root == null) {
            return root;
        }
        if(root.val < L) {
            return trimBST(root.right, L, R);
        }
        if(root.val > R) {
            return trimBST(root.left, L, R);
        }
        
        root.left = trimBST(root.left, L, R);
        root.right = trimBST(root.right, L, R);

        return root;
    }
}
//Design a data structure that supports all following operations in average O(1) time.

//insert(val): Inserts an item val to the set if not already present.
//remove(val): Removes an item val from the set if present.
//getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.

//Example:
// Init an empty set.
//RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
//randomSet.insert(1);

// Returns false as 2 does not exist in the set.
//randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
//randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
//randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
//randomSet.remove(1);

// 2 was already in the set, so return false.
//randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
//randomSet.getRandom();

class RandomizedSet {
    HashMap<Integer, Integer> map;
    ArrayList<Integer> values;

    /** Initialize your data structure here. */
    public RandomizedSet() {
        map = new HashMap<Integer, Integer>();
        values = new ArrayList<Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(!map.containsKey(val)) {
            map.put(val, val);
            values.add(val);
            return true;
        }
        else {
            return false;
        }
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(map.containsKey(val)) {
            map.remove(val);
            values.remove(values.indexOf(val));
            return true;
        }
        return false;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int random = (int)(Math.random() * values.size());
        int valueToReturn = values.get(random);
        return map.get(valueToReturn);
    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */

// Given two 1d vectors, implement an iterator to return their elements alternately.

// For example, given two 1d vectors:

// v1 = [1, 2]
// v2 = [3, 4, 5, 6]
// By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1, 3, 2, 4, 5, 6].

// Follow up: What if you are given k 1d vectors? How well can your code be extended to such cases?

/**
 * Your ZigzagIterator object will be instantiated and called as such: 
 * ZigzagIterator i = new ZigzagIterator(v1, v2);
 * while (i.hasNext()) v[f()] = i.next();
 */

public class ZigZagIterator {
    private Iterator<Integer> i;
    private Iterator<Integer> j;
    private Iterator<Integer> temp;

    public ZigzagIterator(List<Integer> v1, List<Integer> v2) {
        i = v1.iterator();
        j = v2.iterator();
    }

    public int next() {
        if(i.hasNext()) {
            temp = i;
            i = j;
            j = temp;
        }
        
        return j.next();
    }

    public boolean hasNext() {
        return i.hasNext() || j.hasNext();
    }
}
//Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
//push(x) -- Push element x onto stack.
//pop() -- Removes the element on top of the stack.
//top() -- Get the top element.
//getMin() -- Retrieve the minimum element in the stack.

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
class MinStack {
    class Node {
        int data;
        int min;
        Node next;
        
        public Node(int data, int min) {
            this.data = data;
            this.min = min;
            this.next = null;
        }
    }
    Node head;
    
    /** initialize your data structure here. */
    public MinStack() {
        
    }
    
    public void push(int x) {
        if(head == null) {
            head = new Node(x, x);
        } else {
            Node newNode = new Node(x, Math.min(x, head.min));
            newNode.next = head;
            head = newNode;
        }
    }
    
    public void pop() {
        head = head.next;
    }
    
    public int top() {
        return head.data;
    }
    
    public int getMin() {
        return head.min;
    }
}
// Given a non-negative integer represented as non-empty a singly linked list of digits, plus one to the integer.

// You may assume the integer do not contain any leading zero, except the number 0 itself.

// The digits are stored such that the most significant digit is at the head of the list.

// Example:
// Input:
// 1->2->3

// Output:
// 1->2->4

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class PlusOneLinkedList {
    public ListNode plusOne(ListNode head) {
        if(plusOneRecursive(head) == 0) {
            return head;
        } else {
            ListNode newHead = new ListNode(1);
            newHead.next = head;
            
            return newHead;
        }
    }
    
    private int plusOneRecursive(ListNode head) {
        if(head == null) {
            return 1;
        }
        
        int carry = plusOneRecursive(head.next);
        
        if(carry == 0) {
            return 0;
        }
        
        int value = head.val + 1;
        head.val = value % 10;

        return value/10;
    }
}
//Given a linked list, determine if it has a cycle in it.
//Follow up:
//Can you solve it without using extra space?
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if(head == null || head.next == null) {
            return false;
        }
        
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast != null && fast.next != null && fast != slow) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return fast == slow;
    }
}

// You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

// You may assume the two numbers do not contain any leading zero, except the number 0 itself.

// Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
// Output: 7 -> 0 -> 8

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class AddTwoNumbers {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode current1 = l1;
        ListNode current2 = l2;
        
        ListNode head = new ListNode(0);
        ListNode currentHead = head;
        
        int sum = 0;
        
        while(current1 != null || current2 != null) {
            sum /= 10;
            
            if(current1 != null) {
                sum += current1.val;
                current1 = current1.next;
            }
            
            if(current2 != null) {
                sum += current2.val;
                current2 = current2.next;
            }
            
            currentHead.next = new ListNode(sum % 10);
            currentHead = currentHead.next;
        }
        
        
        if(sum / 10 == 1) {
            currentHead.next = new ListNode(1);
        }
        
        return head.next;
    }
}
// Reverse a singly linked list.

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class ReverseLinkedList {
    public ListNode reverseList(ListNode head) {
        if(head == null) {
            return head;
        }
    
        ListNode newHead = null;
        
        while(head != null) {
            ListNode next = head.next;
            head.next = newHead;
            newHead = head;
            head = next;
        }
        
        return newHead;
    }
}
// Write a function to delete a node (except the tail) in a singly linked list, given only access to that node.

// Supposed the linked list is 1 -> 2 -> 3 -> 4 and you are given the third node with value 3, the linked list should become 1 -> 2 -> 4 after calling your function.

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class DeleteNodeInALinkedList {
    public void deleteNode(ListNode node) {
        node.val = node.next.val;
        node.next = node.next.next;
    }
}
// Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class MergeKSortedLists {
    public ListNode mergeKLists(ListNode[] lists) {
        if (lists==null||lists.length==0) {
            return null;
        }
        
        PriorityQueue<ListNode> queue= new PriorityQueue<ListNode>(lists.length,new Comparator<ListNode>(){
            @Override
            public int compare(ListNode o1,ListNode o2){
                if (o1.val<o2.val) {
                    return -1;
                } else if (o1.val==o2.val) {
                    return 0;
                } else {
                    return 1;
                }
            }
        });
        
        ListNode dummy = new ListNode(0);
        ListNode tail=dummy;
        
        for (ListNode node:lists) {
            if (node!=null) {
                queue.add(node);
            }
        }

        while (!queue.isEmpty()){
            tail.next=queue.poll();
            tail=tail.next;

            if (tail.next!=null) {
                queue.add(tail.next);
            }
        }

        return dummy.next;
    }
}
/**
 * Definition for singly-linked list.
 * public class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) { val = x; }
 * }
 */
public class PalindromeLinkedList {
    public boolean isPalindrome(ListNode head) {
        if(head == null || head.next == null) {
            return true;
        }
        
        Stack<Integer> stack = new Stack<Integer>();
        
        ListNode fast = head;
        ListNode slow = head;

        while(fast != null && fast.next != null) {
            stack.push(slow.val);
            fast = fast.next.next;
            slow = slow.next;
        }
        
        if(fast != null) {
            slow = slow.next;
        }
        
        while(slow != null) {
            if(stack.pop() != slow.val) {
                return false;
            }

            slow = slow.next;
        }
        
        return true;
    }
}
// Given a binary tree

//     struct TreeLinkNode {
//       TreeLinkNode *left;
//       TreeLinkNode *right;
//       TreeLinkNode *next;
//     }
// Populate each next pointer to point to its next right node. If there is no next right node, the next pointer should be set to NULL.

// Initially, all next pointers are set to NULL.

// Note:

// You may only use constant extra space.
// You may assume that it is a perfect binary tree (ie, all leaves are at the same level, and every parent has two children).
// For example,
// Given the following perfect binary tree,
//          1
//        /  \
//       2    3
//      / \  / \
//     4  5  6  7
// After calling your function, the tree should look like:
//          1 -> NULL
//        /  \
//       2 -> 3 -> NULL
//      / \  / \
//     4->5->6->7 -> NULL

/**
 * Definition for binary tree with next pointer.
 * public class TreeLinkNode {
 *     int val;
 *     TreeLinkNode left, right, next;
 *     TreeLinkNode(int x) { val = x; }
 * }
 */
public class PopulatingNextRightPointersInEachNode {
    public void connect(TreeLinkNode root) {
        if(root == null) {
            return;
        }
        
        Queue<TreeLinkNode> queue = new LinkedList<TreeLinkNode>();
        
        queue.add(root);
        
        while(!queue.isEmpty()) {
            Queue<TreeLinkNode> currentLevel = new LinkedList<TreeLinkNode>();
            
            TreeLinkNode temp = null;
            
            while(!queue.isEmpty()) {
                TreeLinkNode current = queue.remove();
                current.next = temp;
                temp = current;
                
                
                if(current.right != null) {
                    currentLevel.add(current.right);
                }
                
                if(current.left!= null) {
                    currentLevel.add(current.left);
                }
            }
            
            queue = currentLevel;
        }
    }
}
// Given a binary tree, determine if it is height-balanced.

// For this problem, a height-balanced binary tree is defined as a binary tree in which the depth of the two subtrees of every node never differ by more than 1.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class BalancedBinaryTree {
    boolean balanced = true;
    
    public boolean isBalanced(TreeNode root) {
        height(root);
        return balanced;
    }
    
    private int height(TreeNode root) {
        if(root == null) {
            return 0;
        }
        
        int leftHeight = height(root.left);
        int rightHeight = height(root.right);
        
        if(Math.abs(leftHeight - rightHeight) > 1) {
            balanced = false;
        }
        
        return 1 + Math.max(leftHeight, rightHeight);
    }
}
// Given two binary trees, write a function to check if they are equal or not.

// Two binary trees are considered equal if they are structurally identical and the nodes have the same value.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class SameTree {
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if(p == null && q == null) {
            return true;
        }
        
        if(p == null && q != null || q == null && p != null) {
            return false;
        }
        
        if(p.val != q.val) {
            return false;
        }
        
        return isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }
}
// Given an array where elements are sorted in ascending order, convert it to a height balanced BST.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class ConvertSortedArrayToBinarySearchTree {
    public TreeNode sortedArrayToBST(int[] nums) {
        if(nums.length == 0) {
            return null;
        }
        
        TreeNode root = helper(nums, 0, nums.length - 1);
        
        return root;
    }
    
    private TreeNode helper(int[] nums, int start, int end) {
        if(start <= end) {
            int mid = (start + end) / 2;
            
            TreeNode current = new TreeNode(nums[mid]);
            
            current.left = helper(nums, start, mid - 1);
            current.right = helper(nums, mid + 1, end);
            
            return current;
        }
        
        return null;
    }
}
// Given an 2D board, count how many battleships are in it. The battleships are represented with 'X's, empty slots are represented with '.'s. You may assume the following rules:

// You receive a valid board, made of only battleships or empty slots.
// Battleships can only be placed horizontally or vertically. In other words, they can only be made of the shape 1xN (1 row, N columns) or Nx1 (N rows, 1 column), where N can be of any size.
// At least one horizontal or vertical cell separates between two battleships - there are no adjacent battleships.

// Example:
// X..X
// ...X
// ...X
// In the above board there are 2 battleships.

// Invalid Example:
// ...X
// XXXX
// ...X
// This is an invalid board that you will not receive - as battleships will always have a cell separating between them.

// Follow up:
// Could you do it in one-pass, using only O(1) extra memory and without modifying the value of the board?

public class BattleshipsInABoard {
    public int countBattleships(char[][] board) {
        int ships = 0;
        
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(board[i][j] == 'X') {
                    ships++;
                    sink(board, i, j, 1);
                }
            }
        }
        
        return ships;
    }
    
    public void sink(char[][] board, int i, int j, int numberOfShips) {
        if(i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] == '.') {
            return;
        }
        
        board[i][j] = '.';
        sink(board, i + 1, j, numberOfShips + 1);
        sink(board, i, j + 1, numberOfShips + 1);
    }
}
// Given a binary tree, find its maximum depth.

// The maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf node.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class MaximumDepthOfABinaryTree {
    public int maxDepth(TreeNode root) {
        if(root == null) {
            return 0;
        }
        
        return 1 + Math.max(maxDepth(root.left), maxDepth(root.right));
    }
}
// Given a 2d grid map of '1's (land) and '0's (water), count the number of islands. An island is surrounded by water and is formed by connecting adjacent lands horizontally or vertically. You may assume all four edges of the grid are all surrounded by water.

// Example 1:

// 11110
// 11010
// 11000
// 00000
// Answer: 1

// Example 2:

// 11000
// 11000
// 00100
// 00011
// Answer: 3

public class NumberOfIslands {
    char[][] gridCopy;
    
    public int numIslands(char[][] grid) {
        //set grid copy to the current grid
        gridCopy = grid;
        
        //initialize number of islands to zero
        int numberOfIslands = 0;
        
        //iterate through every index of the grid
        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[0].length; j++) {
                //attempt to "sink" the current index of the grid
                numberOfIslands += sink(gridCopy, i, j);
            }
        }
        
        //return the total number of islands
        return numberOfIslands;
    }
    
    int sink(char[][] grid, int i, int j) {
        //check the bounds of i and j and if the current index is an island or not (1 or 0)
        if(i < 0 || i >= grid.length || j < 0 || j >= grid[0].length || grid[i][j] == '0') {
            return 0;
        }
        
        //set current index to 0
        grid[i][j] = '0';
        
        // sink all neighbors of current index
        sink(grid, i + 1, j);
        sink(grid, i - 1, j);
        sink(grid, i, j + 1);
        sink(grid, i, j - 1);
        
        //increment number of islands
        return 1;
    }
}
//There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.
//Example:
//Given n = 3. 

//At first, the three bulbs are [off, off, off].
//After first round, the three bulbs are [on, on, on].
//After second round, the three bulbs are [on, off, on].
//After third round, the three bulbs are [on, off, off]. 

//So you should return 1, because there is only one bulb is on.

class BulbSwitcher {
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n);
    }
}

// Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), find the minimum number of conference rooms required.

// For example,
// Given [[0, 30],[5, 10],[15, 20]],
// return 2.

/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class MeetingRoomsII {
    public int minMeetingRooms(Interval[] intervals) {
        int[] starts = new int[intervals.length];
        int[] ends = new int[intervals.length];

        for(int i=0; i<intervals.length; i++) {
            starts[i] = intervals[i].start;
            ends[i] = intervals[i].end;
        }

        Arrays.sort(starts);
        Arrays.sort(ends);

        int rooms = 0;
        int endsItr = 0;

        for(int i=0; i<starts.length; i++) {
            if(starts[i]<ends[endsItr]) {
                rooms++;
            } else {
                endsItr++;
            }
        }

        return rooms;
    }
}
// Given an array of meeting time intervals consisting of start and end times [[s1,e1],[s2,e2],...] (si < ei), determine if a person could attend all meetings.

// For example,
// Given [[0, 30],[5, 10],[15, 20]],
// return false.

/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class MeetingRooms {
    public boolean canAttendMeetings(Interval[] intervals) {
        if(intervals == null) {
            return false;
        }

        // Sort the intervals by start time
        Arrays.sort(intervals, new Comparator<Interval>() {
            public int compare(Interval a, Interval b) { return a.start - b.start; }
        });

        for(int i = 1; i < intervals.length; i++) {
            if(intervals[i].start < intervals[i - 1].end) {
                return false;
            }
        }

        return true;
    }
}
// Find the contiguous subarray within an array (containing at least one number) which has the largest sum.

// For example, given the array [-2,1,-3,4,-1,2,1,-5,4],
// the contiguous subarray [4,-1,2,1] has the largest sum = 6.

public class Solution {

    public int maxSubArray(int[] nums) {
        
        int[] dp = new int[nums.length];
        
        dp[0] = nums[0];
        
        int max = dp[0];
        
        for(int i = 1; i < nums.length; i++) {
            
            dp[i] = nums[i] + (dp[i - 1] > 0 ? dp[i - 1] : 0);
            
            max = Math.max(dp[i], max);
            
        }
        
        return max;
        
    }

}//Given an array of integers and an integer k, find out whether there are two distinct indices i and 
//j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

class ContainsDuplicatesII {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i = 0; i < nums.length; i++) {
            int current = nums[i];
            if(map.containsKey(current) && i - map.get(current) <= k) {
                return true;
            } else {
                map.put(current, i);
            }
        }
        
        return false;
    }
}
//Design a data structure that supports all following operations in average O(1) time.

//insert(val): Inserts an item val to the set if not already present.
//remove(val): Removes an item val from the set if present.
//getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.

//Example:
// Init an empty set.
//RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
//randomSet.insert(1);

// Returns false as 2 does not exist in the set.
//randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
//randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
//randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
//randomSet.remove(1);

// 2 was already in the set, so return false.
//randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
//randomSet.getRandom();

class RandomizedSet {
    HashMap<Integer, Integer> map;
    ArrayList<Integer> values;

    /** Initialize your data structure here. */
    public RandomizedSet() {
        map = new HashMap<Integer, Integer>();
        values = new ArrayList<Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(!map.containsKey(val)) {
            map.put(val, val);
            values.add(val);
            return true;
        }
        else {
            return false;
        }
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(map.containsKey(val)) {
            map.remove(val);
            values.remove(values.indexOf(val));
            return true;
        }
        return false;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int random = (int)(Math.random() * values.size());
        int valueToReturn = values.get(random);
        return map.get(valueToReturn);
    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */

// Given an unsorted array nums, reorder it in-place such that nums[0] <= nums[1] >= nums[2] <= nums[3]....

// For example, given nums = [3, 5, 2, 1, 6, 4], one possible answer is [1, 6, 2, 5, 3, 4].

public class WiggleSort {
    public void wiggleSort(int[] nums) {
        for(int i = 1; i < nums.length; i++) {
            int current = nums[i - 1];
            
            if((i % 2 == 1) == (current > nums[i])) {
                nums[i - 1] = nums[i];
                nums[i] = current;
            }
        }
    }
}
// According to the Wikipedia's article: "The Game of Life, also known simply as Life, is a cellular automaton devised by the British mathematician John Horton Conway in 1970."

// Given a board with m by n cells, each cell has an initial state live (1) or dead (0). Each cell interacts with its eight neighbors (horizontal, vertical, diagonal) using the following four rules (taken from the above Wikipedia article):

    // Any live cell with fewer than two live neighbors dies, as if caused by under-population.
    // Any live cell with two or three live neighbors lives on to the next generation.
    // Any live cell with more than three live neighbors dies, as if by over-population..
    // Any dead cell with exactly three live neighbors becomes a live cell, as if by reproduction.
    // Write a function to compute the next state (after one update) of the board given its current state.

// Follow up: 
// Could you solve it in-place? Remember that the board needs to be updated at the same time: You cannot update some cells first and then use their updated values to update other cells.
// In this question, we represent the board using a 2D array. In principle, the board is infinite, which would cause problems when the active area encroaches the border of the array. How would you address these problems?

public class GameOfLife {
    public void gameOfLife(int[][] board) {
        if(board == null || board.length == 0) {
            return;
        }
        
        int m = board.length;
        int n = board[0].length;
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                int lives = liveNeighbors(board, m, n, i, j);
                
                if(board[i][j] ==  1 && lives >= 2 && lives <= 3) {
                    board[i][j] = 3;
                }
                
                if(board[i][j] == 0 && lives == 3) {
                    board[i][j] = 2;
                }
            }
        }
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                board[i][j] >>= 1;
            }
        }
    }
    
    private int liveNeighbors(int[][] board, int m, int n, int i, int j) {
        int lives = 0;
        
        for(int x = Math.max(i - 1, 0); x <= Math.min(i + 1, m - 1); x++) {
            for(int y = Math.max(j - 1, 0); y <= Math.min(j + 1, n - 1); y++) {
                lives += board[x][y] & 1;
            }
        }
        
        lives -= board[i][j] & 1;
        
        return lives;
    }
}
// Given a sorted integer array without duplicates, return the summary of its ranges.

// For example, given [0,1,2,4,5,7], return ["0->2","4->5","7"].

public class SummaryRanges {
    public List<String> summaryRanges(int[] nums) {
        List<String> result = new ArrayList();
        
        if(nums.length == 1) {
            result.add(nums[0] + "");
            return result;
        }
        
        for(int i = 0; i < nums.length; i++) {
            int current = nums[i];
            
            while(i + 1 < nums.length && (nums[i + 1] - nums[i] == 1)) {
                i++;
            }
            
            if(current != nums[i]) {
                result.add(current + "->" + nums[i]);
            } else {
                result.add(current + "");
            }
        }
        
        return result;
    }
}
// Given an integer n, generate a square matrix filled with elements from 1 to n2 in spiral order.

// For example,
// Given n = 3,

// You should return the following matrix:
// [
//  [ 1, 2, 3 ],
//  [ 8, 9, 4 ],
//  [ 7, 6, 5 ]
// ]

public class SpiralMatrix {
    public int[][] generateMatrix(int n) {
        int[][] spiral = new int[n][n];
        
        if(n == 0) {
            return spiral;
        }
        
        int rowStart = 0;
        int colStart = 0;
        int rowEnd = n - 1;
        int colEnd = n -1;
        int number = 1;
        
        while(rowStart <= rowEnd && colStart <= colEnd) {
            for(int i = colStart; i <= colEnd; i++) {
                spiral[rowStart][i] = number++;
            }
            
            rowStart++;
            
            for(int i = rowStart; i <= rowEnd; i++) {
                spiral[i][colEnd] = number++;
            }
            
            colEnd--;
            
            for(int i = colEnd; i >= colStart; i--) {
                if(rowStart <= rowEnd) {
                    spiral[rowEnd][i] = number++;
                }
            }
            
            rowEnd--;
            
            for(int i = rowEnd; i >= rowStart; i--) {
                if(colStart <= colEnd) {
                    spiral[i][colStart] = number++;
                }
            }
            
            colStart++;
        }
        
        return spiral;
    }
}
    // Given a collection of integers that might contain duplicates, nums, return all possible subsets.

// Note: The solution set must not contain duplicate subsets.

// For example,
// If nums = [1,2,2], a solution is:

// [
//   [2],
//   [1],
//   [1,2,2],
//   [2,2],
//   [1,2],
//   []
// ]

public class SubsetsII {
    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);
        
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        
        if(nums.length == 0 || nums == null) {
            return result;
        }
        
        helper(nums, new ArrayList<Integer>(), 0, result);
        
        return result;
    }
    
    
    public void helper(int[] nums, ArrayList<Integer> current, int index, List<List<Integer>> result) {
        result.add(current);
        
        for(int i = index; i < nums.length; i++) {
            if(i > index && nums[i] == nums[i - 1]) {
                continue;
            }
            
            ArrayList<Integer> newCurrent = new ArrayList<Integer>(current);
            newCurrent.add(nums[i]);
            helper(nums, newCurrent, i + 1, result);
        }
    }
}
//A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
//
//The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
//
//How many possible unique paths are there?

class UniquePaths {
    public int uniquePaths(int m, int n) {
        Integer[][] map = new Integer[m][n];
        
        //only 1 way to get to ith row, 0th column (move down)
        for(int i = 0; i < m; i++){
            map[i][0] = 1;
        }
        
        //only 1 way to get to ith column, 0th row (move right)
        for(int j= 0; j < n; j++){
            map[0][j]=1;
        }
        
        //x ways to get to ith row, jth column (# of ways to get to
        //ith - 1 row, jth column + # of ways to get to jth - 1 column
        //ith column
        for(int i = 1;i < m; i++){
            for(int j = 1; j < n; j++){
                map[i][j] = map[i - 1][j] + map[i][j - 1];
            }
        }

        return map[m - 1][n - 1];
    }
}
// Find the contiguous subarray within an array (containing at least one number) which has the largest product.

// For example, given the array [2,3,-2,4],
// the contiguous subarray [2,3] has the largest product = 6.

public class MaximumProductSubarray {
    public int maxProduct(int[] nums) {
        if(nums == null || nums.length == 0) {
            return 0;
        }
        
        int result = nums[0];
        int max = nums[0];
        int min = nums[0];
        
        for(int i = 1; i < nums.length; i++) {
            int temp = max;
            max = Math.max(Math.max(nums[i] * max, nums[i] * min), nums[i]);
            min = Math.min(Math.min(nums[i] * temp, nums[i] * min), nums[i]);
            
            if(max > result) {
                result = max;
            }
        }
        
        return result;
    }
}
// Given a set of distinct integers, nums, return all possible subsets.

// Note: The solution set must not contain duplicate subsets.

// For example,
// If nums = [1,2,3], a solution is:

// [
//   [3],
//   [1],
//   [2],
//   [1,2,3],
//   [1,3],
//   [2,3],
//   [1,2],
//   []
// ]

public class Subsets {
    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> result = new ArrayList<>();
        
        recurse(result, nums, new Stack<>(), 0);
        
        return result;
    }
    
    private void recurse(List<List<Integer>> result, int[] nums, Stack path, int position) {
        if(position == nums.length) {
            result.add(new ArrayList<>(path));
            return;
        }

        path.push(nums[position]);

        recurse(result, nums, path, position + 1);
        
        path.pop();
        
        recurse(result, nums, path, position + 1);
    }
}
//Given an array and a value, remove all instances of that value in-place and return the new length.
//Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
//The order of elements can be changed. It doesn't matter what you leave beyond the new length.

//Example:
//Given nums = [3,2,2,3], val = 3,
//Your function should return length = 2, with the first two elements of nums being 2.

class RemoveElement {
    public int removeElement(int[] nums, int val) {
        int index = 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != val) {
                nums[index++] = nums[i];
            }
        }
        
        return index;
    }
}
// Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

// (i.e., 0 1 2 4 5 6 7 might become 4 5 6 7 0 1 2).

// You are given a target value to search. If found in the array return its index, otherwise return -1.

// You may assume no duplicate exists in the array.

public class SearchInRotatedSortedArray {
    public int search(int[] nums, int target) {
        int left = 0;
        int right = nums.length - 1;
        
        while(left <= right) {
            int mid = left + (right - left) / 2;
            
            if(nums[mid] == target) {
                return mid;
            }
            
            if(nums[left] <= nums[mid]) {
                if(target < nums[mid] && target >= nums[left]) {
                    right = mid - 1;
                } else {
                    left = mid + 1;
                }
            }
            
            if(nums[mid] <= nums[right]) {
                if(target > nums[mid] && target <= nums[right]) {
                    left = mid + 1;
                } else {
                    right = mid - 1;
                }
            }
        }
        
        return -1;
    }
}
// Given a collection of intervals, merge all overlapping intervals.

// For example,
// Given [1,3],[2,6],[8,10],[15,18],
// return [1,6],[8,10],[15,18].

/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
class MergeIntervals {
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> result = new ArrayList<Interval>();
        if(intervals == null || intervals.size() == 0) {
            return result;
        }
        
        Interval[] allIntervals = intervals.toArray(new Interval[intervals.size()]);
        Arrays.sort(allIntervals, new Comparator<Interval>() {
           public int compare(Interval a, Interval b) {
               if(a.start == b.start) {
                   return a.end - b.end;
               }
               return a.start - b.start;
           } 
        });
        
        for(Interval i: allIntervals) {
            if (result.size() == 0 || result.get(result.size() - 1).end < i.start) {
                    result.add(i);
            } else {
                result.get(result.size() - 1).end = Math.max(result.get(result.size() - 1).end, i.end);
            }
        }
        
        return result;
    }
}
//On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).
//
//Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.
//
//Example 1:
//Input: cost = [10, 15, 20]
//Output: 15
//Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
//Example 2:
//Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
//Output: 6
//Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
//Note:
//cost will have a length in the range [2, 1000].
//Every cost[i] will be an integer in the range [0, 999].

class MinCostClimbingStairs {
    public int minCostClimbingStairs(int[] cost) {
        if(cost == null || cost.length == 0) {
            return 0;
        }
        if(cost.length == 1) {
            return cost[0];
        }
        if(cost.length == 2) {
            return Math.min(cost[0], cost[1]);
        }
        
        int[] dp = new int[cost.length];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for(int i = 2; i < cost.length; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i], dp[i - 2] + cost[i]);
        }
        
        return Math.min(dp[cost.length - 1], dp[cost.length -2]);
    }
}

//Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
//
//The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
//
//You may assume the integer does not contain any leading zero, except the number 0 itself.
//
//Example 1:
//
//Input: [1,2,3]
//Output: [1,2,4]
//Explanation: The array represents the integer 123.
//Example 2:
//
//Input: [4,3,2,1]
//Output: [4,3,2,2]
//Explanation: The array represents the integer 4321.

class Solution {
    public int[] plusOne(int[] digits) {
        for(int i = digits.length - 1; i >= 0; i--) {
            if(digits[i] < 9) {
                digits[i]++;
                return digits;
            }

            digits[i] = 0;
        }

        int[] result = new int[digits.length + 1];
        result[0] = 1;

        return result;
    }
}
// Say you have an array for which the ith element is the price of a given stock on day i.

// If you were only permitted to complete at most one transaction (ie, buy one and sell one share of the stock), design an algorithm to find the maximum profit.

// Example 1:
// Input: [7, 1, 5, 3, 6, 4]
// Output: 5

// max. difference = 6-1 = 5 (not 7-1 = 6, as selling price needs to be larger than buying price)
// Example 2:
// Input: [7, 6, 4, 3, 1]
// Output: 0

// In this case, no transaction is done, i.e. max profit = 0.

public class BestTimeToBuyAndSellStock {
    public int maxProfit(int[] prices) {
        //Kadane's algorithm
        if(prices.length == 0) {
            return 0;
        }
        
        int min = prices[0];
        int max = 0;
        
        for(int i = 1; i < prices.length; i++) {
            if(prices[i] > min) {
                max = Math.max(max, prices[i] - min);
            } else {
                min = prices[i];
            }
        }
        
        return max;
    }
}
// You are given an n x n 2D matrix representing an image.

// Rotate the image by 90 degrees (clockwise).

// Follow up:
    // Could you do this in-place?

public class RotateImage {
    public void rotate(int[][] matrix) {
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < i; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = temp;
            }
        }
        
        for(int i = 0; i < matrix.length; i++) {
            for(int j = 0; j < matrix[0].length / 2; j++) {
                int temp = matrix[i][j];
                matrix[i][j] = matrix[i][matrix[0].length - 1 - j];
                matrix[i][matrix[0].length - 1 - j] = temp;
            }
        }
    }
}
// Given a set of non-overlapping intervals, insert a new interval into the intervals (merge if necessary).

// You may assume that the intervals were initially sorted according to their start times.

// Example 1:
// Given intervals [1,3],[6,9], insert and merge [2,5] in as [1,5],[6,9].

// Example 2:
// Given [1,2],[3,5],[6,7],[8,10],[12,16], insert and merge [4,9] in as [1,2],[3,10],[12,16].

// This is because the new interval [4,9] overlaps with [3,5],[6,7],[8,10].

/**
 * Definition for an interval.
 * public class Interval {
 *     int start;
 *     int end;
 *     Interval() { start = 0; end = 0; }
 *     Interval(int s, int e) { start = s; end = e; }
 * }
 */
public class InsertInterval {
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        int i = 0;

        while(i < intervals.size() && intervals.get(i).end < newInterval.start) {
            i++;
        }

        while(i < intervals.size() && intervals.get(i).start <= newInterval.end) {
            newInterval = new Interval(Math.min(intervals.get(i).start, newInterval.start), Math.max(intervals.get(i).end, newInterval.end));
            intervals.remove(i);
        }
        
        intervals.add(i, newInterval);

        return intervals;
    }
}
// Given a sorted integer array where the range of elements are in the inclusive range [lower, upper], return its missing ranges.

// For example, given [0, 1, 3, 50, 75], lower = 0 and upper = 99, return ["2", "4->49", "51->74", "76->99"].

public class MissingRanges {
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        ArrayList<String> result = new ArrayList<String>();
        for(int i = 0; i <= nums.length; i++) {
            long start = i == 0 ? lower : (long)nums[i - 1] + 1;
            long end = i == nums.length ? upper : (long)nums[i] - 1;
            addMissing(result, start, end);
        }
        
        return result;
    }
    
    void addMissing(ArrayList<String> result, long start, long end) {
        if(start > end) {
            return;
        } else if(start == end) {
            result.add(start + "");
        } else {
            result.add(start + "->" + end);
        }
    }
}
// Suppose you are at a party with n people (labeled from 0 to n - 1) and among them, there may exist one celebrity. The definition of a celebrity is that all the other n - 1 people know him/her but he/she does not know any of them.

// Now you want to find out who the celebrity is or verify that there is not one. The only thing you are allowed to do is to ask questions like: "Hi, A. Do you know B?" to get information of whether A knows B. You need to find out the celebrity (or verify there is not one) by asking as few questions as possible (in the asymptotic sense).

// You are given a helper function bool knows(a, b) which tells you whether A knows B. Implement a function int findCelebrity(n), your function should minimize the number of calls to knows.

// Note: There will be exactly one celebrity if he/she is in the party. Return the celebrity's label if there is a celebrity in the party. If there is no celebrity, return -1.

/* The knows API is defined in the parent class Relation.
      boolean knows(int a, int b); */

public class FindTheCelebrity extends Relation {
    public int findCelebrity(int n) {
        //initialize candidate to 0
        int candidate = 0;
        
        //find viable candidate
        for(int i = 1; i < n; i++) {
            if(knows(candidate, i)) {
                candidate = i;
            }
        }
        
        //check that everyone else knows the candidate
        for(int i = 0; i < n; i++) {
            //if the candidate knows the current person or the current person does not know the candidate, return -1 (candidate is not a celebrity)
            if(i != candidate && knows(candidate, i) || !knows(i, candidate)) {
                return -1;
            }
        }
        
        //return the celebrity
        return candidate;
    }
}
// Given an array of n integers where n > 1, nums, return an array output such that output[i] is equal to the product of all the elements of nums except nums[i].

// Solve it without division and in O(n).

// For example, given [1,2,3,4], return [24,12,8,6].

// Follow up:
// Could you solve it with constant space complexity? (Note: The output array does not count as extra space for the purpose of space complexity analysis.)

public class ProductOfArrayExceptSelf {
    public int[] productExceptSelf(int[] nums) {
        int n = nums.length;
        int[] result = new int[n];
        int left = 1;
        
        for(int i = 0; i < nums.length; i++) {
            if(i > 0) {
                left *= nums[i - 1];
            }
            
            result[i] = left;
        }
        
        int right = 1;
        
        for(int i = n - 1; i >= 0; i--) {
            if(i < n - 1) {
                right *= nums[i + 1];
            }
            
            result[i] *= right;
        }
        
        return result;
    }
}
//Given a matrix of m x n elements (m rows, n columns), return all elements of the matrix in spiral order.
//
//Example 1:
//
//Input:
//[
 //[ 1, 2, 3 ],
 //[ 4, 5, 6 ],
 //[ 7, 8, 9 ]
//]
//Output: [1,2,3,6,9,8,7,4,5]
//Example 2:
//
//Input:
//[
  //[1, 2, 3, 4],
  //[5, 6, 7, 8],
  //[9,10,11,12]
//]
//Output: [1,2,3,4,8,12,11,10,9,5,6,7]

class SpiralMatrix {
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> result = new ArrayList<Integer>();
        if(matrix == null || matrix.length == 0) {
            return result;
        }
        
        int rowStart = 0;
        int rowEnd = matrix.length - 1;
        int colStart = 0;
        int colEnd = matrix[0].length - 1;
        while(rowStart <= rowEnd && colStart <= colEnd) {
            for(int i = colStart; i <= colEnd; i++) {
                result.add(matrix[rowStart][i]);
            }
            rowStart++;
            
            for(int i = rowStart; i <= rowEnd; i++) {
                result.add(matrix[i][colEnd]);
            }
            colEnd--;
            
            if(rowStart <= rowEnd) {
                for(int i = colEnd; i >= colStart; i--) {
                    result.add(matrix[rowEnd][i]);
                }
            }
            rowEnd--;
            
            if(colStart <= colEnd) {
                for(int i = rowEnd; i >= rowStart; i--) {
                    result.add(matrix[i][colStart]);
                }
            }   
            colStart++;
        }
        
        return result;
    }
}
//Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.
//You may assume that the array is non-empty and the majority element always exist in the array.

class MajorityElement {
    public int majorityElement(int[] nums) {
        if(nums.length == 1) {
            return nums[0];
        }
        
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int current: nums) {
            if(map.containsKey(current) && map.get(current) + 1 > nums.length / 2) {
                return current;
            } else if(map.containsKey(current)) {
                map.put(current, map.get(current) + 1);
            } else {
                map.put(current, 1);
            }
        }
        
        //no majority element exists
        return -1;
    }
}
//Given an array of integers where 1 ≤ a[i] ≤ n (n = size of array), some elements appear twice and others appear once.
//
//Find all the elements of [1, n] inclusive that do not appear in this array.
//
//Could you do it without extra space and in O(n) runtime? You may assume the returned list does not count as extra space.
//
//Example:
//
//Input:
//[4,3,2,7,8,2,3,1]
//
//Output:
//[5,6]

class FindAllNumbersDisappearedInAnArray {
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> result = new ArrayList<Integer>();
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i = 1; i <= nums.length; i++) {
            map.put(i, 1);
        }
        
        for(int i = 0; i < nums.length; i++) {
            if(map.containsKey(nums[i])) {
                map.put(nums[i], -1);
            }
        }
        
        for(int i: map.keySet()) {
            if(map.get(i) != -1) {
                result.add(i);
            }
        }
        
        return result;
    }
}
//Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right 
//which minimizes the sum of all numbers along its path.
//Note: You can only move either down or right at any point in time.
//Example 1:
//[[1,3,1],
 //[1,5,1],
 //[4,2,1]]
//Given the above grid map, return 7. Because the path 1→3→1→1→1 minimizes the sum.

class MinimumPathSum {
    public int minPathSum(int[][] grid) {
        for(int i = 1; i < grid.length; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for(int i = 1; i < grid[0].length; i++) {
            grid[0][i] += grid[0][i - 1];
        }
        
        for(int i = 1; i < grid.length; i++) {
            for(int j = 1; j < grid[0].length; j++) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        
        return grid[grid.length - 1][grid[0].length - 1];
    }
}
// Given an unsorted array of integers, find the length of the longest consecutive elements sequence.

// For example,
// Given [100, 4, 200, 1, 3, 2],
// The longest consecutive elements sequence is [1, 2, 3, 4]. Return its length: 4.

// Your algorithm should run in O(n) complexity.

class LongestConsecutiveSequence {
    public int longestConsecutive(int[] nums) {
        if(nums == null || nums.length == 0) {
            return 0;
        }
        
        Set<Integer> set = new HashSet<Integer>();
        for(int n: nums) {
            set.add(n);
        }
        
        int maxLength = 0;
        for(int n: set) {
            if(!set.contains(n - 1)) {
                int current = n;
                int currentMax = 1;
                
                while(set.contains(n + 1)) {
                    currentMax++;
                    n++;
                }
                
                maxLength = Math.max(maxLength, currentMax);
            }
        }
        
        return maxLength;
    }
}
// Given an unsorted array return whether an increasing subsequence of length 3 exists or not in the array.

// Formally the function should:
// Return true if there exists i, j, k 
// such that arr[i] < arr[j] < arr[k] given 0 ≤ i < j < k ≤ n-1 else return false.
// Your algorithm should run in O(n) time complexity and O(1) space complexity.

// Examples:
// Given [1, 2, 3, 4, 5],
// return true.

// Given [5, 4, 3, 2, 1],
// return false.

public class IncreasingTripletSequence {
    public boolean increasingTriplet(int[] nums) {
        int firstMin = Integer.MAX_VALUE;
        int secondMin = Integer.MAX_VALUE;
        
        for(int n : nums) {
            if(n <= firstMin) {
                firstMin = n;
            } else if(n < secondMin) {
                secondMin = n;
            } else if(n > secondMin) {
                return true;
            }
        }
        
        return false;
    }
}
// Given a 2D board and a word, find if the word exists in the grid.

// The word can be constructed from letters of sequentially adjacent cell, where "adjacent" cells are those horizontally or vertically neighboring. The same letter cell may not be used more than once.

// For example,
// Given board =

// [
//   ['A','B','C','E'],
//   ['S','F','C','S'],
//   ['A','D','E','E']
// ]
// word = "ABCCED", -> returns true,
// word = "SEE", -> returns true,
// word = "ABCB", -> returns false.

public class WordSearch {
    public boolean exist(char[][] board, String word) {
        char[] w = word.toCharArray();
        
        for(int i = 0; i < board.length; i++) {
            for(int j = 0; j < board[0].length; j++) {
                if(search(board, i, j, w, 0)) {
                    return true;
                }
            }
        }
        
        return false;
    }
    
    public boolean search(char[][] board, int i, int j, char[] w, int index) {
        if(index == w.length) {
            return true;
        }

        if(i < 0 || i >= board.length || j < 0 || j >= board[0].length) {
            return false;
        }        

        if(board[i][j] != w[index]) {
            return false;
        }
        
        board[i][j] ^= 256;

        boolean exist = search(board, i + 1, j, w, index + 1) ||
                        search(board, i - 1, j, w, index + 1) ||
                        search(board, i, j + 1, w, index + 1) ||
                        search(board, i, j - 1, w, index + 1);
        board[i][j] ^= 256;

        return exist;
    }
}
// You want to build a house on an empty land which reaches all buildings in the shortest amount of distance. You can only move up, down, left and right. You are given a 2D grid of values 0, 1 or 2, where:

// Each 0 marks an empty land which you can pass by freely.
// Each 1 marks a building which you cannot pass through.
// Each 2 marks an obstacle which you cannot pass through.
// For example, given three buildings at (0,0), (0,4), (2,2), and an obstacle at (0,2):

// 1 - 0 - 2 - 0 - 1
// |   |   |   |   |
// 0 - 0 - 0 - 0 - 0
// |   |   |   |   |
// 0 - 0 - 1 - 0 - 0
// The point (1,2) is an ideal empty land to build a house, as the total travel distance of 3+3+1=7 is minimal. So return 7.

// Note:
// There will be at least one building. If it is not possible to build such house according to the above rules, return -1.

public class Shortest {
    public int shortestDistance(int[][] grid) {
        if(grid == null || grid.length == 0 || grid[0].length == 0) {
            return -1;
        }
        
        final int[] shift = {0, 1, 0, -1, 0};
        
        int rows = grid.length;
        int columns = grid[0].length;
        
        int[][] distance = new int[rows][columns];
        int[][] reach = new int[rows][columns];
        
        int numberOfBuildings = 0;
        
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                if(grid[i][j] == 1) {
                    numberOfBuildings++;
                    Queue<int[]> queue = new LinkedList<int[]>();
                    queue.offer(new int[] {i, j});
                    
                    boolean[][] visited = new boolean[rows][columns];
                    
                    int relativeDistance = 1;
                    
                    while(!queue.isEmpty()) {
                        int qSize = queue.size();
                        
                        for(int q = 0; q < qSize; q++) {
                            int[] current = queue.poll();
                            
                            for(int k = 0; k < 4; k++) {
                                int nextRow = current[0] + shift[k];
                                int nextColumn = current[1] + shift[k + 1];
                            
                                if(nextRow >= 0 && nextRow < rows && nextColumn >= 0 && nextColumn < columns && grid[nextRow][nextColumn] == 0 && !visited[nextRow][nextColumn]) {
                                    distance[nextRow][nextColumn] += relativeDistance;
                                    reach[nextRow][nextColumn]++;
                                
                                    visited[nextRow][nextColumn] = true;
                                    queue.offer(new int[] {nextRow, nextColumn});
                                }   
                            }
                        }
                        
                        relativeDistance++;
                    }
                }
            }
        }
    
        int shortest = Integer.MAX_VALUE;
    
        for(int i = 0; i < rows; i++) {
            for(int j = 0; j < columns; j++) {
                if(grid[i][j] == 0 && reach[i][j] == numberOfBuildings) {
                    shortest = Math.min(shortest, distance[i][j]);
                }
            }
        }
    
        return shortest == Integer.MAX_VALUE ? -1 : shortest;
    }
}
// Remove the minimum number of invalid parentheses in order to make the input string valid. Return all possible results.

// Note: The input string may contain letters other than the parentheses ( and ).

// Examples:
// "()())()" -> ["()()()", "(())()"]
// "(a)())()" -> ["(a)()()", "(a())()"]
// ")(" -> [""]

public class RemoveInvalidParentheses {
    public List<String> removeInvalidParentheses(String s) {
        List<String> result = new ArrayList<>();
        remove(s, result, 0, 0, new char[]{'(', ')'});
        return result;
    }

    public void remove(String s, List<String> result, int last_i, int last_j,  char[] par) {
        for (int stack = 0, i = last_i; i < s.length(); i++) {
            if (s.charAt(i) == par[0]) {
                stack++;
            }

            if (s.charAt(i) == par[1]) {
                stack--;
            }

            if (stack >= 0) {
                continue;
            }
            
            for (int j = last_j; j <= i; j++) {
                if (s.charAt(j) == par[1] && (j == last_j || s.charAt(j - 1) != par[1])) {
                    remove(s.substring(0, j) + s.substring(j + 1, s.length()), result, i, j, par);
                }
            }

            return;
        }
        
        String reversed = new StringBuilder(s).reverse().toString();
        
        if (par[0] == '(')  {
            // finished left to right
            remove(reversed, result, 0, 0, new char[]{')', '('});
        } else {
            // finished right to left
            result.add(reversed);
        }
    }
}
// Given a binary tree, check whether it is a mirror of itself (ie, symmetric around its center).

// For example, this binary tree [1,2,2,3,4,4,3] is symmetric:

//     1
//    / \
//   2   2
//  / \ / \
// 3  4 4  3
// But the following [1,2,2,null,3,null,3] is not:
//     1
//    / \
//   2   2
//    \   \
//    3    3

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class SymmetricTree {
    public boolean isSymmetric(TreeNode root) {
        if(root == null) {
            return true;
        }
        
        return helper(root.left, root.right);
    }
    
    public boolean helper(TreeNode left, TreeNode right) {
        if(left == null && right == null) {
            return true;
        }
        
        if(left == null || right == null || left.val != right.val) {
            return false;
        }
        
        return helper(left.right, right.left) && helper(left.left, right.right);
    }
}
// Given an m x n matrix of non-negative integers representing the height of each unit cell in a continent, the "Pacific ocean" touches the left and top edges of the matrix and the "Atlantic ocean" touches the right and bottom edges.

// Water can only flow in four directions (up, down, left, or right) from a cell to another one with height equal or lower.

// Find the list of grid coordinates where water can flow to both the Pacific and Atlantic ocean.

// Note:
    // The order of returned grid coordinates does not matter.
    // Both m and n are less than 150.

// Example:

// Given the following 5x5 matrix:

//   Pacific ~   ~   ~   ~   ~ 
//        ~  1   2   2   3  (5) *
//        ~  3   2   3  (4) (4) *
//        ~  2   4  (5)  3   1  *
//        ~ (6) (7)  1   4   5  *
//        ~ (5)  1   1   2   4  *
//           *   *   *   *   * Atlantic

// Return:

// [[0, 4], [1, 3], [1, 4], [2, 2], [3, 0], [3, 1], [4, 0]] (positions with parentheses in above matrix).

public class PacificAtlanticWaterFlow {
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> result = new LinkedList<>();
        
        //error checking
        if(matrix == null || matrix.length == 0 || matrix[0].length == 0) {
            return result;
        }
        
        int n = matrix.length;
        int m = matrix[0].length;
        
        boolean[][] pacific = new boolean[n][m];
        boolean[][] atlantic = new boolean[n][m];
        
        for(int i = 0; i < n; i++) {
            dfs(matrix, pacific, Integer.MIN_VALUE, i, 0);
            dfs(matrix, atlantic, Integer.MIN_VALUE, i, m - 1);
        }
        
        for(int i = 0; i < m; i++) {
            dfs(matrix, pacific, Integer.MIN_VALUE, 0, i);
            dfs(matrix, atlantic, Integer.MIN_VALUE, n - 1, i);
        }
        
        for(int i = 0; i < n; i++) {
            for(int j = 0; j < m; j++) {
                if(pacific[i][j] && atlantic[i][j]) {
                    result.add(new int[] {i, j});
                }
            }
        }
        
        return result;
    }
    
    public void dfs(int[][] matrix, boolean[][] visited, int height, int x, int y) {
        int n = matrix.length;
        int m = matrix[0].length;
        
        if(x < 0 || x >= n || y < 0 || y >= m || visited[x][y] || matrix[x][y] < height) {
            return;
        }
        
        visited[x][y] = true;
        
        dfs(matrix, visited, matrix[x][y], x + 1, y);
        dfs(matrix, visited, matrix[x][y], x - 1, y);
        dfs(matrix, visited, matrix[x][y], x, y + 1);
        dfs(matrix, visited, matrix[x][y], x, y - 1);
    }
}
// Given a binary tree, return the level order traversal of its nodes' values. (ie, from left to right, level by level).

// For example:
// Given binary tree [3,9,20,null,null,15,7],
//     3
//    / \
//   9  20
//     /  \
//    15   7
// return its level order traversal as:
// [
//   [3],
//   [9,20],
//   [15,7]
// ]

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class BinarySearchTreeLevelOrderTraversal {
    public List<List<Integer>> levelOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<List<Integer>>();
        
        if(root == null) {
            return result;
        }

        Queue<TreeNode> queue = new LinkedList<TreeNode>();
        
        queue.add(root);
        
        List<Integer> tempList = new ArrayList<Integer>();
        tempList.add(root.val);
        result.add(tempList);
        
        while(!queue.isEmpty()) {
            Queue<TreeNode> currentLevel = new LinkedList<TreeNode>();
            
            List<Integer> list = new ArrayList<Integer>();
            
            while(!queue.isEmpty()) {
                TreeNode current = queue.remove();
                
                if(current.left != null) {
                    currentLevel.add(current.left);
                    list.add(current.left.val);
                }
                
                if(current.right != null) {
                    currentLevel.add(current.right);
                    list.add(current.right.val);
                }
            }
            
            if(list.size() > 0) {
                result.add(list);
            }

            queue = currentLevel;
        }
        
        return result;
    }
}
// You are given a m x n 2D grid initialized with these three possible values.

// -1 - A wall or an obstacle.
// 0 - A gate.
// INF - Infinity means an empty room. We use the value 231 - 1 = 2147483647 to represent INF as you may assume that the distance to a gate is less than 2147483647.
// Fill each empty room with the distance to its nearest gate. If it is impossible to reach a gate, it should be filled with INF.

// For example, given the 2D grid:
// INF  -1  0  INF
// INF INF INF  -1
// INF  -1 INF  -1
//   0  -1 INF INF
// After running your function, the 2D grid should be:
//   3  -1   0   1
//   2   2   1  -1
//   1  -1   2  -1
//   0  -1   3   4

public class Solution {
    public void wallsAndGates(int[][] rooms) {
        //iterate through the matrix calling dfs on all indices that contain a zero
        for(int i = 0; i < rooms.length; i++) {
            for(int j = 0; j < rooms[0].length; j++) {
                if(rooms[i][j] == 0) {
                    dfs(rooms, i, j, 0);
                }
            }
        }
    }
    
    void dfs(int[][] rooms, int i, int j, int distance) {
        //if you have gone out of the bounds of the array or you have run into a wall/obstacle, return
        // room[i][j] < distance also ensure that we do not overwrite any previously determined distance if it is shorter than our current distance
        if(i < 0 || i >= rooms.length || j < 0 || j >= rooms[0].length || rooms[i][j] < distance) {
            return;
        }
        
        //set current index's distance to distance
        rooms[i][j] = distance;
        
        //recurse on all adjacent neighbors of rooms[i][j]
        dfs(rooms, i + 1, j, distance + 1);
        dfs(rooms, i - 1, j, distance + 1);
        dfs(rooms, i, j + 1, distance + 1);
        dfs(rooms, i, j - 1, distance + 1);
    }
}
// Clone an undirected graph. Each node in the graph contains a label and a list of its neighbors.

// OJ's undirected graph serialization:
// Nodes are labeled uniquely.

// We use # as a separator for each node, and , as a separator for node label and each neighbor of the node.
// As an example, consider the serialized graph {0,1,2#1,2#2,2}.

// The graph has a total of three nodes, and therefore contains three parts as separated by #.

// First node is labeled as 0. Connect node 0 to both nodes 1 and 2.
// Second node is labeled as 1. Connect node 1 to node 2.
// Third node is labeled as 2. Connect node 2 to node 2 (itself), thus forming a self-cycle.
// Visually, the graph looks like the following:

//        1
//       / \
//      /   \
//     0 --- 2
//          / \
//          \_/

/**
 * Definition for undirected graph.
 * class UndirectedGraphNode {
 *     int label;
 *     List<UndirectedGraphNode> neighbors;
 *     UndirectedGraphNode(int x) { label = x; neighbors = new ArrayList<UndirectedGraphNode>(); }
 * };
 */
public class CloneGraph {
    public HashMap<Integer, UndirectedGraphNode> map = new HashMap<Integer, UndirectedGraphNode>();
    
    public UndirectedGraphNode cloneGraph(UndirectedGraphNode node) {
        if(node == null) {
            return null;
        }
        
        if(map.containsKey(node.label)) {
            return map.get(node.label);
        }
        
        UndirectedGraphNode newNode = new UndirectedGraphNode(node.label);
        map.put(newNode.label, newNode);
        
        for(UndirectedGraphNode neighbor : node.neighbors) {
            newNode.neighbors.add(cloneGraph(neighbor));
        }
        
        return newNode;
    }
}
//Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
//
//For example, given n = 3, a solution set is:
//
//[
  //"((()))",
  //"(()())",
  //"(())()",
  //"()(())",
  //"()()()"
//]

class GenerateParentheses {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<String>();
        generateParenthesisRecursive(result, "", 0, 0, n);
        
        return result;
    }
    
    public void generateParenthesisRecursive(List<String> result, String current, int open, int close, int n) {
        if(current.length() == n * 2) {
            result.add(current);
            return;
        }
        
        if(open < n) {
            generateParenthesisRecursive(result, current + "(", open + 1, close, n);
        }
        
        if(close < open) {
            generateParenthesisRecursive(result, current + ")", open, close + 1, n);
        }
    }
}
// Given an Android 3x3 key lock screen and two integers m and n, where 1 ≤ m ≤ n ≤ 9, count the total number of unlock patterns of the Android lock screen, which consist of minimum of m keys and maximum n keys.

// Rules for a valid pattern:
    // Each pattern must connect at least m keys and at most n keys.
    // All the keys must be distinct.
    // If the line connecting two consecutive keys in the pattern passes through any other keys, the other keys must have previously selected in the pattern. No jumps through non selected key is allowed.
    // The order of keys used matters.

// Explanation:
// | 1 | 2 | 3 |
// | 4 | 5 | 6 |
// | 7 | 8 | 9 |
// Invalid move: 4 - 1 - 3 - 6 
// Line 1 - 3 passes through key 2 which had not been selected in the pattern.

// Invalid move: 4 - 1 - 9 - 2
// Line 1 - 9 passes through key 5 which had not been selected in the pattern.

// Valid move: 2 - 4 - 1 - 3 - 6
// Line 1 - 3 is valid because it passes through key 2, which had been selected in the pattern

// Valid move: 6 - 5 - 4 - 1 - 9 - 2
// Line 1 - 9 is valid because it passes through key 5, which had been selected in the pattern.

// Example:
// Given m = 1, n = 1, return 9.

public class AndroidUnlockPatterns {
    public int numberOfPatterns(int m, int n) {
        //initialize a 10x10 matrix
        int skip[][] = new int[10][10];
        
        //initialize indices of skip matrix (all other indices in matrix are 0 by default)
        skip[1][3] = skip[3][1] = 2;
        skip[1][7] = skip[7][1] = 4;
        skip[3][9] = skip[9][3] = 6;
        skip[7][9] = skip[9][7] = 8;
        skip[1][9] = skip[9][1] = skip[2][8] = skip[8][2] = skip[3][7] = skip [7][3] = skip[6][4] = skip[4][6] = 5;
        
        //initialize visited array
        boolean visited[] = new boolean[10];
        
        //initialize total number to 0
        int totalNumber = 0;
        
        //run DFS for each length from m to n
        for(int i = m; i <= n; ++i) {
            totalNumber += DFS(visited, skip, 1, i - 1) * 4; //1, 3, 7, and 9 are symmetric so multiply this result by 4
            totalNumber += DFS(visited, skip, 2, i - 1) * 4; //2, 4, 6, and 8 are symmetric so multiply this result by 4
            totalNumber += DFS(visited, skip, 5, i - 1); //do not multiply by 4 because 5 is unique         
        }
        
        return totalNumber;
    }
    
    int DFS(boolean visited[], int[][] skip, int current, int remaining) {
        //base cases
        if(remaining < 0) {
            return 0;
        }
        
        if(remaining == 0) {
            return 1;
        }
        
        //mark the current node as visited
        visited[current] = true;
        
        //initialize total number to 0
        int totalNumber = 0;
        
        for(int i = 1; i <= 9; ++i) {
            //if the current node has not been visited and (two numbers are adjacent or skip number has already been visited)
            if(!visited[i] && (skip[current][i] == 0 || visited[skip[current][i]])) {
                totalNumber += DFS(visited, skip, i, remaining - 1);
            }
        }
        
        //mark the current node as not visited
        visited[current] = false;
        
        //return total number
        return totalNumber;
    }
}
// Write a function to generate the generalized abbreviations of a word.

// Example:
// Given word = "word", return the following list (order does not matter):
// ["word", "1ord", "w1rd", "wo1d", "wor1", "2rd", "w2d", "wo2", "1o1d", "1or1", "w1r1", "1o2", "2r1", "3d", "w3", "4"]

public class GeneralizedAbbreviation {
    public List<String> generateAbbreviations(String word) {
        List<String> result = new ArrayList<String>();
        
        backtrack(result, word, 0, "", 0);
        
        return result;
    }
    
    void backtrack(List result, String word, int position, String current, int count) {
        if(position == word.length()) {
            if(count > 0) {
                current += count;   
            }
            
            result.add(current);
        } else {
            backtrack(result, word, position + 1, current, count + 1);
            backtrack(result, word, position + 1, current + (count > 0 ? count : "") + word.charAt(position), 0);
        }
    }
}
//Given a collection of distinct numbers, return all possible permutations.
//
//For example,
//[1,2,3] have the following permutations:
//[
  //[1,2,3],
  //[1,3,2],
  //[2,1,3],
  //[2,3,1],
  //[3,1,2],
  //[3,2,1]
//]

class Permutations {
    public List<List<Integer>> permute(int[] nums) {
        LinkedList<List<Integer>> result = new LinkedList<List<Integer>>();
        result.add(new ArrayList<Integer>());
        for (int n: nums) {
            int size = result.size();
            while(size > 0) {
                List<Integer> current = result.pollFirst();
                for (int i = 0; i <= current.size(); i++) {
                    List<Integer> temp = new ArrayList<Integer>(current);
                    temp.add(i, n);
                    result.add(temp);
                }
                size--;
            }
        }

        return result;
    }
}
// Given a digit string, return all possible letter combinations that the number could represent.

// A mapping of digit to letters (just like on the telephone buttons) is given below.

// 2 - abc
// 3 - def
// 4 - ghi
// 5 - jkl
// 6 - mno
// 7 - pqrs
// 8 - tuv
// 9 - wxyz

// Input:Digit string "23"
// Output: ["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

class LetterCombinationsOfAPhoneNumber {
    public List<String> letterCombinations(String digits) {
        List<String> result = new ArrayList<String>();
        
        if(digits == null || digits.length() == 0) {
            return result;
        }
        
        String[] mapping = {
            "0",
            "1",
            "abc",
            "def",
            "ghi",
            "jkl",
            "mno",
            "pqrs",
            "tuv",
            "wxyz"
        };
        
        letterCombinationsRecursive(result, digits, "", 0, mapping);
        
        return result;
    }
    
    public void letterCombinationsRecursive(List<String> result, String digits, String current, int index, String[] mapping) {
        if(index == digits.length()) {
            result.add(current);
            return;
        }
        
        String letters = mapping[digits.charAt(index) - '0'];
        for(int i = 0; i < letters.length(); i++) {
            letterCombinationsRecursive(result, digits, current + letters.charAt(i), index + 1, mapping);
        }
    }
}
//Given an integer, write a function to determine if it is a power of two.
//
//Example 1:
//
//Input: 1
//Output: true
//Example 2:
//
//Input: 16
//Output: true
//Example 3:
//
//Input: 218
//Output: false

class PowerOfTwo {
    public boolean isPowerOfTwo(int n) {
        long i = 1;
        while(i < n) {
            i <<= 1;
        }
        
        return i == n;
    }
}
//TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl 
//and it returns a short URL such as http://tinyurl.com/4e9iAk.
//
//Design the encode and decode methods for the TinyURL service. There is no restriction on how your 
//encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL 
//and the tiny URL can be decoded to the original URL.

public class EncodeAndDecodeTinyURL {
    HashMap<String, String> map = new HashMap<String, String>();
    String characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int count = 1;

    public String getKey() {
        String key = "";
        while(count > 0) {
            count--;
            key += characters.charAt(count);
            count /= characters.length();
        }
        
        return key;
    }
    
    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        String key = getKey();
        map.put(key, longUrl);
        count++;
            
        return "http://tinyurl.com/" + key;
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        return map.get(shortUrl.replace("http://tinyurl.com/", ""));
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.decode(codec.encode(url));
//Determine whether an integer is a palindrome. Do this without extra space.

class PalindromeNumber {
    public boolean isPalindrome(int x) {
        if(x < 0) {
            return false;
        }
        
        int num = x;
        int reversed = 0;
        
        while(num != 0) {
            reversed = reversed * 10 + num % 10;
            num /= 10;
        }
        
        return x == reversed;
    }
}
//There are 1000 buckets, one and only one of them contains poison, the rest are filled with water. 
//They all look the same. If a pig drinks that poison it will die within 15 minutes. What is the 
//minimum amount of pigs you need to figure out which bucket contains the poison within one hour.

//Answer this question, and write an algorithm for the follow-up general case.

//Follow-up:
//If there are n buckets and a pig drinking poison will die within m minutes, how many pigs (x) 
//you need to figure out the "poison" bucket within p minutes? There is exact one bucket with poison.

class PoorPigs {
    public int poorPigs(int buckets, int minutesToDie, int minutesToTest) {    
        int numPigs = 0;
        while (Math.pow(minutesToTest / minutesToDie + 1, numPigs) < buckets) {
            numPigs++;
        }
        
        return numPigs;
    }
}
//There are n bulbs that are initially off. You first turn on all the bulbs. Then, you turn off every second bulb. On the third round, you toggle every third bulb (turning on if it's off or turning off if it's on). For the ith round, you toggle every i bulb. For the nth round, you only toggle the last bulb. Find how many bulbs are on after n rounds.
//Example:
//Given n = 3. 

//At first, the three bulbs are [off, off, off].
//After first round, the three bulbs are [on, on, on].
//After second round, the three bulbs are [on, off, on].
//After third round, the three bulbs are [on, off, off]. 

//So you should return 1, because there is only one bulb is on.

class BulbSwitcher {
    public int bulbSwitch(int n) {
        return (int)Math.sqrt(n);
    }
}
//Given a non-empty array of digits representing a non-negative integer, plus one to the integer.
//
//The digits are stored such that the most significant digit is at the head of the list, and each element in the array contain a single digit.
//
//You may assume the integer does not contain any leading zero, except the number 0 itself.
//
//Example 1:
//
//Input: [1,2,3]
//Output: [1,2,4]
//Explanation: The array represents the integer 123.
//Example 2:
//
//Input: [4,3,2,1]
//Output: [4,3,2,2]
//Explanation: The array represents the integer 4321.

class Solution {
    public int[] plusOne(int[] digits) {
        for(int i = digits.length - 1; i >= 0; i--) {
            if(digits[i] < 9) {
                digits[i]++;
                return digits;
            }

            digits[i] = 0;
        }

        int[] result = new int[digits.length + 1];
        result[0] = 1;

        return result;
    }
}
//Given a non-negative integer num, repeatedly add all its digits until the result has only one digit.

//For example:
//Given num = 38, the process is like: 3 + 8 = 11, 1 + 1 = 2. Since 2 has only one digit, return it.

//Follow up:
//Could you do it without any loop/recursion in O(1) runtime?

class AddDigits {
    public int addDigits(int num) {
        while(num >= 10) {
            int temp = 0;
            while(num > 0) {
                temp += num % 10;
                num /= 10;
            }
            num = temp;
        }
        
        return num;
    }
}
//You are given coins of different denominations and a total amount of money amount. Write a function to compute the fewest number of coins that you need to make up that amount. If that amount of money cannot be made up by any combination of the coins, return -1.

//Example 1:
//coins = [1, 2, 5], amount = 11
//return 3 (11 = 5 + 5 + 1)

//Example 2:
//coins = [2], amount = 3
//return -1.

//Note:
//You may assume that you have an infinite number of each kind of coin.

class CoinChange {
    public int coinChange(int[] coins, int amount) {
        if(amount < 1) {
            return 0;
        }
        
        return coinChangeRecursive(coins, amount, new int[amount]);
    }
    
    public int coinChangeRecursive(int[] coins, int amount, int[] dp) {
        if(amount < 0) {
            return -1;
        }
        if(amount == 0) {
            return 0;
        }
        if(dp[amount - 1] != 0) {
            return dp[amount - 1];
        }
        
        int min = Integer.MAX_VALUE;
        for(int coin: coins) {
            int result = coinChangeRecursive(coins, amount - coin, dp);
            if(result >= 0 && result < min) {
                min = 1 + result;
            }
        }
        
        dp[amount - 1] = min == Integer.MAX_VALUE ? -1 : min;
        return dp[amount - 1];
    }
}
//Given a string, your task is to count how many palindromic substrings in this string.
//The substrings with different start indexes or end indexes are counted as different substrings 
//even they consist of same characters.

//Example 1:
//Input: "abc"
//Output: 3
//Explanation: Three palindromic strings: "a", "b", "c".
//Example 2:
//Input: "aaa"
//Output: 6
//Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
//Note:
//The input string length won't exceed 1000.

class PalindromicSubstrings {
    int result = 0;
    public int countSubstrings(String s) {
        if(s == null || s.length() == 0) {
            return 0;
        }
        
        for(int i = 0; i < s.length(); i++) {
            extendPalindrome(s, i, i);
            extendPalindrome(s, i, i + 1);
        }
        
        return result;
    }
    
    public void extendPalindrome(String s, int left, int right) {
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            result++;
            left--;
            right++;
        }
    }
}

//A robot is located at the top-left corner of a m x n grid (marked 'Start' in the diagram below).
//
//The robot can only move either down or right at any point in time. The robot is trying to reach the bottom-right corner of the grid (marked 'Finish' in the diagram below).
//
//How many possible unique paths are there?

class UniquePaths {
    public int uniquePaths(int m, int n) {
        Integer[][] map = new Integer[m][n];
        
        //only 1 way to get to ith row, 0th column (move down)
        for(int i = 0; i < m; i++){
            map[i][0] = 1;
        }
        
        //only 1 way to get to ith column, 0th row (move right)
        for(int j= 0; j < n; j++){
            map[0][j]=1;
        }
        
        //x ways to get to ith row, jth column (# of ways to get to
        //ith - 1 row, jth column + # of ways to get to jth - 1 column
        //ith column
        for(int i = 1;i < m; i++){
            for(int j = 1; j < n; j++){
                map[i][j] = map[i - 1][j] + map[i][j - 1];
            }
        }

        return map[m - 1][n - 1];
    }
}

// Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

// Example:
// For num = 5 you should return [0,1,1,2,1,2].

// Follow up:
	// It is very easy to come up with a solution with run time O(n*sizeof(integer)). But can you do it in linear time O(n) /possibly in a single pass?
	// Space complexity should be O(n).
	// Can you do it like a boss? Do it without using any builtin function like __builtin_popcount in c++ or in any other language.

public class CountingBits {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        
        bits[0] = 0;
        
        for(int i = 1; i <= num; i++) {
            bits[i] = bits[i >> 1] + (i & 1);
        }
        
        return bits;
    }
}
//  Given a 2D grid, each cell is either a wall 'W', an enemy 'E' or empty '0' (the number zero), return the maximum enemies you can kill using one bomb.
// The bomb kills all the enemies in the same row and column from the planted point until it hits the wall since the wall is too strong to be destroyed.
// Note that you can only put the bomb at an empty cell.

// Example:
// For the given grid

// 0 E 0 0
// E 0 W E
// 0 E 0 0

// return 3. (Placing a bomb at (1,1) kills 3 enemies)

 public class BombEnemy {
     public int maxKilledEnemies(char[][] grid) {
        if(grid == null || grid.length == 0 ||  grid[0].length == 0) {
            return 0;
        }

        int max = 0;
        int row = 0;
        int[] col = new int[grid[0].length];

        for(int i = 0; i<grid.length; i++) {
            for(int j = 0; j<grid[0].length;j++) {
                if(grid[i][j] == 'W') {
                    continue;
                }

                if(j == 0 || grid[i][j-1] == 'W') {
                     row = killedEnemiesRow(grid, i, j);
                }

                if(i == 0 || grid[i-1][j] == 'W') {
                     col[j] = killedEnemiesCol(grid,i,j);
                }

                if(grid[i][j] == '0') {
                    max = (row + col[j] > max) ? row + col[j] : max;
                }
            }
        }
        
        return max;
    }

    //calculate killed enemies for row i from column j
    private int killedEnemiesRow(char[][] grid, int i, int j) {
        int num = 0;

        while(j <= grid[0].length-1 && grid[i][j] != 'W') {
            if(grid[i][j] == 'E') {
                num++;
            }

            j++;
        }

        return num;
    }

    //calculate killed enemies for  column j from row i
    private int killedEnemiesCol(char[][] grid, int i, int j) {
        int num = 0;

        while(i <= grid.length -1 && grid[i][j] != 'W'){
            if(grid[i][j] == 'E') {
                num++;
            }

            i++;
        }

        return num;
    }
}
// Given a non-empty string s and a dictionary wordDict containing a list of non-empty words, determine if s can be segmented into a space-separated sequence of one or more dictionary words. You may assume the dictionary does not contain duplicate words.

// For example, given
// s = "leetcode",
// dict = ["leet", "code"].

// Return true because "leetcode" can be segmented as "leet code".

public class WordBreak {
    public boolean wordBreak(String s, Set<String> wordDict) {
        boolean[] dp = new boolean[s.length() + 1];
        
        dp[0] = true;
        
        for(int i = 1; i <= s.length(); i++) {
            for(int j = 0; j < i; j++) {
                if(dp[j] && wordDict.contains(s.substring(j, i))) {
                    dp[i] = true;
                    break;
                }
            }
        }
        
        return dp[s.length()];
    }
}
//There are a row of n houses, each house can be painted with one of the three colors: red, blue or green. 
//The cost of painting each house with a certain color is different. You have to paint all the houses such 
//that no two adjacent houses have the same color.

//The cost of painting each house with a certain color is represented by a n x 3 cost matrix. For example, 
//costs[0][0] is the cost of painting house 0 with color red; costs[1][2] is the cost of painting house 1 
//with color green, and so on... Find the minimum cost to paint all houses.

//Note:
//All costs are positive integers.

class PaintHouse {
    public int minCost(int[][] costs) {
        if(costs == null || costs.length == 0) {
            return 0;
        }
        
        for(int i = 1; i < costs.length; i++) {
            costs[i][0] += Math.min(costs[i - 1][1], costs[i - 1][2]);
            costs[i][1] += Math.min(costs[i - 1][0], costs[i - 1][2]);
            costs[i][2] += Math.min(costs[i - 1][0], costs[i - 1][1]);
        }
        
        return Math.min(Math.min(costs[costs.length - 1][0], costs[costs.length - 1][1]), costs[costs.length - 1][2]);
    }
}

//Note: This is an extension of House Robber. (security system is tripped if two ajacent houses are robbed)
//After robbing those houses on that street, the thief has found himself a new place for his thievery so that 
//he will not get too much attention. This time, all houses at this place are arranged in a circle. That means 
//the first house is the neighbor of the last one. Meanwhile, the security system for these houses remain the 
//same as for those in the previous street.
//Given a list of non-negative integers representing the amount of money of each house, determine the maximum 
//amount of money you can rob tonight without alerting the police.

class HouseRobberII {
    public int rob(int[] nums) {
        if (nums.length == 0) {
            return 0;
        }
        if (nums.length < 2) {
            return nums[0];
        }
        
        int[] first = new int[nums.length + 1];
        int[] second = new int[nums.length + 1];
        
        first[0]  = 0;
        first[1]  = nums[0];
        second[0] = 0;
        second[1] = 0;
        
        for (int i = 2; i <= nums.length; i++) {
            first[i] = Math.max(first[i - 1], first[i - 2] + nums[i - 1]);
            second[i] = Math.max(second[i - 1], second[i - 2] + nums[i - 1]);
        }
        
        return Math.max(first[nums.length - 1], second[nums.length]);
    }
}
//On a staircase, the i-th step has some non-negative cost cost[i] assigned (0 indexed).
//
//Once you pay the cost, you can either climb one or two steps. You need to find minimum cost to reach the top of the floor, and you can either start from the step with index 0, or the step with index 1.
//
//Example 1:
//Input: cost = [10, 15, 20]
//Output: 15
//Explanation: Cheapest is start on cost[1], pay that cost and go to the top.
//Example 2:
//Input: cost = [1, 100, 1, 1, 1, 100, 1, 1, 100, 1]
//Output: 6
//Explanation: Cheapest is start on cost[0], and only step on 1s, skipping cost[3].
//Note:
//cost will have a length in the range [2, 1000].
//Every cost[i] will be an integer in the range [0, 999].

class MinCostClimbingStairs {
    public int minCostClimbingStairs(int[] cost) {
        if(cost == null || cost.length == 0) {
            return 0;
        }
        if(cost.length == 1) {
            return cost[0];
        }
        if(cost.length == 2) {
            return Math.min(cost[0], cost[1]);
        }
        
        int[] dp = new int[cost.length];
        dp[0] = cost[0];
        dp[1] = cost[1];
        for(int i = 2; i < cost.length; i++) {
            dp[i] = Math.min(dp[i - 1] + cost[i], dp[i - 2] + cost[i]);
        }
        
        return Math.min(dp[cost.length - 1], dp[cost.length -2]);
    }
}

// Given n, how many structurally unique BST's (binary search trees) that store values 1...n?

// For example,
// Given n = 3, there are a total of 5 unique BST's.

//    1         3     3      2      1
//     \       /     /      / \      \
//      3     2     1      1   3      2
//     /     /       \                 \
//    2     1         2                 3

public class UniqueBinarySearchTree {
    public int numTrees(int n) {
        int[] dp = new int[n + 1];
        
        dp[0] = 1;
        dp[1] = 1;
        
        for(int i = 2; i <= n; i++) {
            for(int j = 1; j <= i; j++) {
                dp[i] += dp[i - j] * dp[j - 1];
            }
        }
        
        return dp[n];
    }
}
// Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

// You have the following 3 operations permitted on a word:

// a) Insert a character
// b) Delete a character
// c) Replace a character

public class EditDistance {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        
        int[][] dp = new int[m + 1][n + 1];

        for(int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        
        for(int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(word1.charAt(i) == word2.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    int a = dp[i][j];
                    int b = dp[i][j + 1];
                    int c = dp[i + 1][j];
                    
                    dp[i + 1][j + 1] = Math.min(a, Math.min(b, c));
                    dp[i + 1][j + 1]++;
                }
            }
        }
        
        return dp[m][n];
    }
}
// You are climbing a stair case. It takes n steps to reach to the top.

// Each time you can either climb 1 or 2 steps. In how many distinct ways can you climb to the top?

// Note: Given n will be a positive integer.

public class ClimbingStairs {
    public int climbStairs(int n) {
        int[] dp = new int[n + 1];
        
        dp[0] = 1;
        dp[1] = 1;
        
        for(int i = 2; i < dp.length; i++) {
            dp[i] = dp[i - 1] + dp[i - 2];
        }
        
        return dp[dp.length - 1];
    } 
}
// Implement regular expression matching with support for '.' and '*'.

// '.' Matches any single character.
// '*' Matches zero or more of the preceding element.

// The matching should cover the entire input string (not partial).

// The function prototype should be:
// bool isMatch(const char *s, const char *p)

// Some examples:
// isMatch("aa","a") → false
// isMatch("aa","aa") → true
// isMatch("aaa","aa") → false
// isMatch("aa", "a*") → true
// isMatch("aa", ".*") → true
// isMatch("ab", ".*") → true
// isMatch("aab", "c*a*b") → true

public class RegularExpressionMatching {
    public boolean isMatch(String s, String p) {
        if(s == null || p == null) {
            return false;
        }
        
        boolean[][] dp = new boolean[s.length() + 1][p.length() + 1];
        dp[0][0] = true;
        
        for(int i = 0; i < p.length(); i++) {
            if(p.charAt(i) == '*' && dp[0][i - 1]) {
                dp[0][i + 1] = true;
            }
        }
        
        for(int i = 0; i < s.length(); i++) {
            for(int j = 0; j < p.length(); j++) {
                if(p.charAt(j) == '.') {
                    dp[i + 1][j + 1] = dp[i][j];
                }
                
                if(p.charAt(j) == s.charAt(i)) {
                    dp[i + 1][j + 1] = dp[i][j];
                }
                
                if(p.charAt(j) == '*') {
                    if(p.charAt(j - 1) != s.charAt(i) && p.charAt(j - 1) != '.') {
                        dp[i + 1][j + 1] = dp[i + 1][j - 1];
                    } else {
                        dp[i + 1][j + 1] = (dp[i + 1][j] || dp[i][j + 1] || dp[i + 1][j - 1]);
                    }
                }
            }
        }
        
        return dp[s.length()][p.length()];
    }
}
// You are a professional robber planning to rob houses along a street. Each house has a certain amount of money stashed, the only constraint stopping you from robbing each of them is that adjacent houses have security system connected and it will automatically contact the police if two adjacent houses were broken into on the same night.

// Given a list of non-negative integers representing the amount of money of each house, determine the maximum amount of money you can rob tonight without alerting the police.

public class Solution {
    public int rob(int[] nums) {
        if(nums.length == 0) {
            return 0;
        }

        if(nums.length == 1) {
            return nums[0];
        }
        
        int[] dp = new int[nums.length];
        
        dp[0] = nums[0];
        dp[1] = nums[0] > nums[1] ? nums[0] : nums[1];

        for(int i = 2; i < nums.length; i++) {
            dp[i] = Math.max(dp[i - 2] + nums[i], dp[i - 1]);
        }
        
        return dp[dp.length - 1];
    }
}
    // There are a row of n houses, each house can be painted with one of the k colors. The cost of painting each house with a certain color is different. You have to paint all the houses such that no two adjacent houses have the same color.

// The cost of painting each house with a certain color is represented by a n x k cost matrix. For example, costs[0][0] is the cost of painting house 0 with color 0; costs[1][2] is the cost of painting house 1 with color 2, and so on... Find the minimum cost to paint all houses.

// Note:
// All costs are positive integers.

// Follow up:
// Could you solve it in O(nk) runtime?

public class PaintHouseII {
    public int minCostII(int[][] costs) {
        if(costs == null|| costs.length == 0) {
            return 0;
        }
        
        int m = costs.length;
        int n = costs[0].length;
        
        int min1 = -1;
        int min2 = -1;
        
        for(int i = 0; i < m; i++) {
            int last1 = min1;
            int last2 = min2;

            min1 = -1;
            min2 = -1;
            
            for(int j = 0; j < n; j++) {
                if(j != last1) {
                    costs[i][j] += last1 < 0 ? 0 : costs[i - 1][last1];
                } else {
                    costs[i][j] += last2 < 0 ? 0 : costs[i - 1][last2];
                }

                if(min1 < 0 || costs[i][j] < costs[i][min1]) {
                    min2 = min1;
                    min1 = j;
                } else if(min2 < 0 || costs[i][j] < costs[i][min2]) {
                    min2 = j;
                }
            }
        }
        
        return costs[m - 1][min1];       
    }
}
// Given a rows x cols screen and a sentence represented by a list of non-empty words, find how many times the given sentence can be fitted on the screen.

// Note:
    // A word cannot be split into two lines.
    // The order of words in the sentence must remain unchanged.
    // Two consecutive words in a line must be separated by a single space.
    // Total words in the sentence won't exceed 100.
    // Length of each word is greater than 0 and won't exceed 10.
    // 1 ≤ rows, cols ≤ 20,000.

// Example 1:

// Input:
// rows = 2, cols = 8, sentence = ["hello", "world"]

// Output: 
// 1

// Explanation:
// hello---
// world---

// The character '-' signifies an empty space on the screen.
// Example 2:

// Input:
// rows = 3, cols = 6, sentence = ["a", "bcd", "e"]

// Output: 
// 2

// Explanation:
// a-bcd- 
// e-a---
// bcd-e-

// The character '-' signifies an empty space on the screen.
// Example 3:

// Input:
// rows = 4, cols = 5, sentence = ["I", "had", "apple", "pie"]

// Output: 
// 1

// Explanation:
// I-had
// apple
// pie-I
// had--

// The character '-' signifies an empty space on the screen.

public class SentenceScreenFitting {
    public int wordsTyping(String[] sentence, int rows, int cols) {
        String s = String.join(" ", sentence) + " ";
        int start = 0;
        int l = s.length();

        for(int i = 0; i < rows; i++) {
            start += cols;
            
            if(s.charAt(start % l) == ' ') {
                start++;
            } else {
                while(start > 0 && s.charAt((start - 1) % l) != ' ') {
                    start--;
                }
            }
        }
        
        return start / s.length();
    }
}
// There is a fence with n posts, each post can be painted with one of the k colors.

// You have to paint all the posts such that no more than two adjacent fence posts have the same color.

// Return the total number of ways you can paint the fence.

// Note:
// n and k are non-negative integers.

public class PaintFence {
    public int numWays(int n, int k) {
        if(n <= 0) {
            return 0;
        }
        
        int sameColorCounts = 0;
        int differentColorCounts = k;
        
        for(int i = 2; i <= n; i++) {
            int temp = differentColorCounts;
            differentColorCounts = (sameColorCounts + differentColorCounts) * (k - 1);
            sameColorCounts = temp;
        }
        
        return sameColorCounts + differentColorCounts;
    }
}
//Given a m x n grid filled with non-negative numbers, find a path from top left to bottom right 
//which minimizes the sum of all numbers along its path.
//Note: You can only move either down or right at any point in time.
//Example 1:
//[[1,3,1],
 //[1,5,1],
 //[4,2,1]]
//Given the above grid map, return 7. Because the path 1→3→1→1→1 minimizes the sum.

class MinimumPathSum {
    public int minPathSum(int[][] grid) {
        for(int i = 1; i < grid.length; i++) {
            grid[i][0] += grid[i - 1][0];
        }
        for(int i = 1; i < grid[0].length; i++) {
            grid[0][i] += grid[0][i - 1];
        }
        
        for(int i = 1; i < grid.length; i++) {
            for(int j = 1; j < grid[0].length; j++) {
                grid[i][j] += Math.min(grid[i - 1][j], grid[i][j - 1]);
            }
        }
        
        return grid[grid.length - 1][grid[0].length - 1];
    }
}

//Given an unsorted array of integers, find the length of longest increasing subsequence.

//For example,
//Given [10, 9, 2, 5, 3, 7, 101, 18],
//The longest increasing subsequence is [2, 3, 7, 101], therefore the length is 4. Note that there may be more than one LIS combination, it is only necessary for you to return the length.

//Your algorithm should run in O(n2) complexity.

//Follow up: Could you improve it to O(n log n) time complexity?

class LongestIncreasingSubsequence {
    public int lengthOfLIS(int[] nums) {
        if(nums == null || nums.length < 1) {
            return 0;
        }

        int[] dp = new int[nums.length];
        dp[0] = 1;
        
        int max = 1;
        for(int i = 1; i < dp.length; i++) {
            int currentMax = 0;
            for(int j = 0; j < i; j++) {
                if(nums[i] > nums[j]) {
                    currentMax = Math.max(currentMax, dp[j]);
                }
            }
            dp[i] = 1 + currentMax;
            max = Math.max(max, dp[i]);
        }

        return max;
    }
}
// Given an integer array with all positive numbers and no duplicates, find the number of possible combinations that add up to a positive integer target.

// Example:

// nums = [1, 2, 3]
// target = 4

// The possible combination ways are:
// (1, 1, 1, 1)
// (1, 1, 2)
// (1, 2, 1)
// (1, 3)
// (2, 1, 1)
// (2, 2)
// (3, 1)

// Note that different sequences are counted as different combinations.

// Therefore the output is 7.

// Follow up:
    // What if negative numbers are allowed in the given array?
    // How does it change the problem?
    // What limitation we need to add to the question to allow negative numbers?

public class CombinationSumIV {
    public int combinationSum4(int[] nums, int target) {
        int[] dp = new int[target + 1];
        dp[0] = 1;
        
        for(int i = 1; i < dp.length; i++) {
            for(int j = 0; j < nums.length; j++) {
                if(i - nums[j] >= 0) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }
        
        return dp[target];
    }
}
// Given a sorted array, remove the duplicates in place such that each element appear only once and return the new length.

// Do not allocate extra space for another array, you must do this in place with constant memory.

// For example,
// Given input array nums = [1,1,2],

// Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the new length.

public class RemoveDuplicatesFromSortedArray {
    public int removeDuplicates(int[] nums) {
        if(nums.length == 0 || nums == null) {
            return 0;
        }

        if(nums.length < 2) {
            return nums.length;
        }
        
        int index = 1;
        
        for(int i = 1; i < nums.length; i++) {
            if(nums[i] != nums[i - 1]) {
                nums[index++] = nums[i];
            }
        }
        
        return index;
    }
}
// Given an array nums, write a function to move all 0's to the end of it while maintaining the relative order of the non-zero elements.

// For example, given nums = [0, 1, 0, 3, 12], after calling your function, nums should be [1, 3, 12, 0, 0].

// Note:
    // You must do this in-place without making a copy of the array.
    // Minimize the total number of operations.

public class MoveZeros {
    public void moveZeroes(int[] nums) {
        if(nums == null || nums.length == 0) {
            return;
        }
        
        int index = 0;
        for(int num : nums) {
            if(num != 0) {
                nums[index] = num;
                index++;
            }
        }
        
        while(index < nums.length) {
            nums[index] = 0;
            index++;
        }
    }
}
//Given a linked list, determine if it has a cycle in it.
//Follow up:
//Can you solve it without using extra space?
/**
 * Definition for singly-linked list.
 * class ListNode {
 *     int val;
 *     ListNode next;
 *     ListNode(int x) {
 *         val = x;
 *         next = null;
 *     }
 * }
 */
public class Solution {
    public boolean hasCycle(ListNode head) {
        if(head == null || head.next == null) {
            return false;
        }
        
        ListNode slow = head;
        ListNode fast = head.next;
        while(fast != null && fast.next != null && fast != slow) {
            slow = slow.next;
            fast = fast.next.next;
        }
        
        return fast == slow;
    }
}
//Given an array and a value, remove all instances of that value in-place and return the new length.
//Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
//The order of elements can be changed. It doesn't matter what you leave beyond the new length.

//Example:
//Given nums = [3,2,2,3], val = 3,
//Your function should return length = 2, with the first two elements of nums being 2.

class RemoveElement {
    public int removeElement(int[] nums, int val) {
        int index = 0;
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] != val) {
                nums[index++] = nums[i];
            }
        }
        
        return index;
    }
}

// Write a function that takes a string as input and returns the string reversed.

// Example:
// Given s = "hello", return "olleh".

public class ReverseString {
    public String reverseString(String s) {
        if(s == null || s.length() == 1 || s.length() == 0) {
            return s;
        }
        
        char[] word = s.toCharArray();
        
        for(int i = 0, j = s.length() - 1; i < j; i++, j--) {
            char temp = word[i];
            word[i] = word[j];
            word[j] = temp;
        }
        
        return new String(word);
    }
}
// Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

// Note:
// You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2. The number of elements initialized in nums1 and nums2 are m and n respectively.

public class MergeSortedArray {
    public void merge(int[] A, int m, int[] B, int n) {
        int i = m - 1;
        int j = n - 1;
        int k = m + n - 1;
        
        while(i >= 0 && j >= 0) {
            A[k--] = A[i] > B[j] ? A[i--] : B[j--];
        }
        
        while(j >= 0) {
            A[k--] = B[j--];
        }
    }
}
// Given an array of n integers nums and a target, find the number of index triplets i, j, k with 0 <= i < j < k < n that satisfy the condition nums[i] + nums[j] + nums[k] < target.

// For example, given nums = [-2, 0, 1, 3], and target = 2.

// Return 2. Because there are two triplets which sums are less than 2:

// [-2, 0, 1]
// [-2, 0, 3]

// Follow up:
    // Could you solve it in O(n2) runtime?

public class 3SumSmaller {
    public int threeSumSmaller(int[] nums, int target) {
        //initialize total count to zero
        int count = 0;
        
        //sort the array
        Arrays.sort(nums);
        
        //loop through entire array
        for(int i = 0; i < nums.length - 2; i++) {
            //set left to i + 1
            int left = i + 1;
            
            //set right to end of array
            int right = nums.length - 1;
            
            //while left index < right index
            while(left < right) {
                //if the 3 indices add to less than the target increment count
                if(nums[i] + nums[left] + nums[right] < target) {
                    //increment the count by the distance between left and right because the array is sorted
                    count += right - left;
                    
                    //decrement right pointer
                    left++;
                } else {
                    //if they sum to a value greater than target...
                    //increment left pointer
                    right--;
                }
            }
        }
        
        return count;
    }
}
// Given an array of n positive integers and a positive integer s, find the minimal length of a contiguous subarray of which the sum ≥ s. If there isn't one, return 0 instead.

// For example, given the array [2,3,1,2,4,3] and s = 7,
// the subarray [4,3] has the minimal length under the problem constraint.

public class MinimumSizeSubarraySum {
    public int minSubArrayLen(int s, int[] nums) {
        if(nums == null || nums.length == 0) {
            return 0;
        }
        
        int i = 0;
        int j = 0;
        int result = Integer.MAX_VALUE;
        int total = 0;
        
        while(i < nums.length) {
            total += nums[i++];
            
            while(total >= s) {
                result = Math.min(result, i - j);
                total -= nums[j++];
            }
        }
        
        return result == Integer.MAX_VALUE ? 0 : result;
    }
}
// Given an array with n objects colored red, white or blue, sort them so that objects of the same color are adjacent, with the colors in the order red, white and blue.

// Here, we will use the integers 0, 1, and 2 to represent the color red, white, and blue respectively.

// Note:
    // You are not suppose to use the library's sort function for this problem.

public class SortColors {
    public void sortColors(int[] nums) {
        int wall = 0;
        
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] < 1) {
                int temp = nums[i];
                nums[i] = nums[wall];
                nums[wall] = temp;
                wall++;
            }
        }
        
        for(int i = 0; i < nums.length; i++) {
            if(nums[i] == 1) {
                int temp = nums[i];
                nums[i] = nums[wall];
                nums[wall] = temp;
                wall++;
            }
        }
    }
}
// Given an array S of n integers, are there elements a, b, c in S such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

// Note: The solution set must not contain duplicate triplets.

// For example, given array S = [-1, 0, 1, 2, -1, -4],

// A solution set is:
// [
//   [-1, 0, 1],
//   [-1, -1, 2]
// ]

public class 3Sum {
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>>  result = new ArrayList<>();

        Arrays.sort(nums);

        for(int i = 0; i < nums.length - 2; i++) {
            if(i > 0 && nums[i] == nums[i - 1]) {
                continue;
            }

            int j = i + 1;
            int k = nums.length - 1;
            int target = -nums[i];

            while(j < k) {
                if(nums[j] + nums[k] == target) {
                    ArrayList<Integer> temp = new ArrayList<Integer>();

                    temp.add(nums[i]);
                    temp.add(nums[j]);
                    temp.add(nums[k]);

                    result.add(temp);

                    j++;
                    k--;

                    while(j < k && nums[j] == nums[j - 1]) {
                        j++;
                    }

                    while(j < k && nums[k] == nums[k + 1]) {
                        k--;
                    }
                } else if(nums[j] + nums[k] > target) {
                    k--;
                } else {
                    j++;
                }
            }
        }

        return result;
    }
}
// Design a data structure that supports the following two operations:

// void addWord(word)
// bool search(word)
// search(word) can search a literal word or a regular expression string containing only letters a-z or .. A . means it can represent any one letter.

// For example:

// addWord("bad")
// addWord("dad")
// addWord("mad")
// search("pad") -> false
// search("bad") -> true
// search(".ad") -> true
// search("b..") -> true

// Note:
    // You may assume that all words are consist of lowercase letters a-z.

public class AddAndSearchWordDataStructure {
    public class TrieNode {
        public TrieNode[] children = new TrieNode[26];
        public String item = "";
    }
    
    private TrieNode root = new TrieNode();

    public void addWord(String word) {
        TrieNode node = root;

        for (char c : word.toCharArray()) {
            if (node.children[c - 'a'] == null) {
                node.children[c - 'a'] = new TrieNode();
            }

            node = node.children[c - 'a'];
        }

        node.item = word;
    }

    public boolean search(String word) {
        return match(word.toCharArray(), 0, root);
    }
    
    private boolean match(char[] chs, int k, TrieNode node) {
        if (k == chs.length) {
            return !node.item.equals(""); 
        }

        if (chs[k] != '.') {
            return node.children[chs[k] - 'a'] != null && match(chs, k + 1, node.children[chs[k] - 'a']);
        } else {
            for (int i = 0; i < node.children.length; i++) {
                if (node.children[i] != null) {
                    if (match(chs, k + 1, node.children[i])) {
                        return true;
                    }
                }
            }
        }

        return false;
    }
}
// Implement a trie with insert, search, and startsWith methods.

// Note:
// You may assume that all inputs are consist of lowercase letters a-z.

// Your Trie object will be instantiated and called as such:
// Trie trie = new Trie();
// trie.insert("somestring");
// trie.search("key");

class TrieNode {

    HashMap<Character, TrieNode> map;
    char character;
    boolean last;
    
    // Initialize your data structure here.
    public TrieNode(char character) {
        
        this.map = new HashMap<Character, TrieNode>();
        this.character = character;
        this.last = false;
        
    }

}

public class ImplementTrie {
    private TrieNode root;

    public Trie() {
        root = new TrieNode(' ');
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        TrieNode current = root;
            
        for(char c : word.toCharArray()) {
            if(!current.map.containsKey(c)) {
                current.map.put(c, new TrieNode(c));
            }
            
            current = current.map.get(c);
        }
        
        current.last = true;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
        TrieNode current = root;
        
        for(char c : word.toCharArray()) {
            if(!current.map.containsKey(c)) {
                return false;
            }
            
            current = current.map.get(c);
        }
        
        if(current.last == true) {
            return true;
        } else {
            return false;
        }
    }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String prefix) {
        TrieNode current = root;
        
        for(char c : prefix.toCharArray()) {
            if(!current.map.containsKey(c)) {
                return false;
            }
            
            current = current.map.get(c);
        }
        
        return true;
    }
}

// Given a set of words (without duplicates), find all word squares you can build from them.

// A sequence of words forms a valid word square if the kth row and column read the exact same string, where 0 ≤ k < max(numRows, numColumns).

// For example, the word sequence ["ball","area","lead","lady"] forms a word square because each word reads the same both horizontally and vertically.

// b a l l
// a r e a
// l e a d
// l a d y

// Note:
    // There are at least 1 and at most 1000 words.
    // All words will have the exact same length.
    // Word length is at least 1 and at most 5.
    // Each word contains only lowercase English alphabet a-z.

public class WordSquares {
    public List<List<String>> wordSquares(String[] words) {
        List<List<String>> ret = new ArrayList<List<String>>();

        if(words.length==0 || words[0].length()==0) {
            return ret;
        }

        Map<String, Set<String>> map = new HashMap<>();

        int squareLen = words[0].length();

        // create all prefix
        for(int i=0;i<words.length;i++){
            for(int j=-1;j<words[0].length();j++){
                if(!map.containsKey(words[i].substring(0, j+1))) {
                    map.put(words[i].substring(0, j+1), new HashSet<String>());
                }

                map.get(words[i].substring(0, j+1)).add(words[i]);
            }
        }

        helper(ret, new ArrayList<String>(), 0, squareLen, map);

        return ret;
    }

    public void helper(List<List<String>> ret, List<String> cur, int matched, int total, Map<String, Set<String>> map){
        if(matched == total) {
            ret.add(new ArrayList<String>(cur));
            return;
        }

        // build search string
        StringBuilder sb = new StringBuilder();

        for(int i=0;i<=matched-1;i++) {
            sb.append(cur.get(i).charAt(matched));
        }

        // bachtracking
        Set<String> cand = map.get(sb.toString());

        if(cand==null) {
            return;
        }

        for(String str:cand){
            cur.add(str);
            helper(ret, cur, matched+1, total, map);
            cur.remove(cur.size()-1);
        }
    }
}
//Say you have an array for which the ith element is the price of a given stock on day i.
//Design an algorithm to find the maximum profit. You may complete as many transactions as you 
//like (ie, buy one and sell one share of the stock multiple times). However, you may not engage 
//in multiple transactions at the same time (ie, you must sell the stock before you buy again).

class BestTimeToBuyAndSellStockII {
    public int maxProfit(int[] prices) {
        if(prices == null || prices.length == 0) {
            return 0;
        }
        
        int profit = 0;
        for(int i = 0; i < prices.length - 1; i++) {
            if(prices[i] < prices[i + 1]) {
                profit += prices[i + 1] - prices[i];
            }
        }
        
        return profit;
    }
}
// Given a stream of integers and a window size, calculate the moving average of all integers in the sliding window.

// For example,
// MovingAverage m = new MovingAverage(3);
// m.next(1) = 1
// m.next(10) = (1 + 10) / 2
// m.next(3) = (1 + 10 + 3) / 3
// m.next(5) = (10 + 3 + 5) / 3

/**
 * Your MovingAverage object will be instantiated and called as such:
 * MovingAverage obj = new MovingAverage(size);
 * double param_1 = obj.next(val);
 */

public class MovingAverageFromDataStream {
    double previousSum = 0.0;
    int maxSize;
    Queue<Integer> window;

    /** Initialize your data structure here. */
    public MovingAverage(int size) {
        this.maxSize = size;
        window = new LinkedList<Integer>();
    }
    
    public double next(int val) {
        if(window.size() == maxSize) {
            previousSum -= window.remove();
        }
        
        window.add(val);
        previousSum += val;

        return previousSum / window.size();
    }
}
// Given a string that contains only digits 0-9 and a target value, return all possibilities to add binary operators (not unary) +, -, or * between the digits so they evaluate to the target value.

// Examples: 
// "123", 6 -> ["1+2+3", "1*2*3"] 
// "232", 8 -> ["2*3+2", "2+3*2"]
// "105", 5 -> ["1*0+5","10-5"]
// "00", 0 -> ["0+0", "0-0", "0*0"]
// "3456237490", 9191 -> []

public class ExpressionAddOperators {
    public List<String> addOperators(String num, int target) {
        List<String> result = new ArrayList<String>();

        if(num == null || num.length() == 0) {
            return result;
        }

        helper(result, "", num, target, 0, 0, 0);
        return result;
    }
    
    public void helper(List<String> result, String path, String num, int target, int pos, long eval, long multed) {
        if(pos == num.length()) {
            if(eval == target) {
                result.add(path);
            }
            
            return;
        }
        
        for(int i = pos; i < num.length(); i++) {
            if(i != pos && num.charAt(pos) == '0') {
                break;
            }

            long cur = Long.parseLong(num.substring(pos, i + 1));

            if(pos == 0) {
                helper(result, path + cur, num, target, i + 1, cur, cur);
            } else {
                helper(result, path + "+" + cur, num, target, i + 1, eval + cur, cur);
                helper(result, path + "-" + cur, num, target, i + 1, eval - cur, -cur);
                helper(result, path + "*" + cur, num, target, i + 1, eval - multed + multed * cur, multed * cur);
            }
        }
    }
}
// Find the kth largest element in an unsorted array. Note that it is the kth largest element in the sorted order, not the kth distinct element.

// For example,
// Given [3,2,1,5,6,4] and k = 2, return 5.

// Note: 
// You may assume k is always valid, 1 ≤ k ≤ array's length.

public class KthLargestElementInAnArray {
    public int findKthLargest(int[] nums, int k) {
        int length = nums.length;
        Arrays.sort(nums);

        return nums[length - k];
    }
}
//Given a string, your task is to count how many palindromic substrings in this string.
//The substrings with different start indexes or end indexes are counted as different substrings 
//even they consist of same characters.

//Example 1:
//Input: "abc"
//Output: 3
//Explanation: Three palindromic strings: "a", "b", "c".
//Example 2:
//Input: "aaa"
//Output: 6
//Explanation: Six palindromic strings: "a", "a", "a", "aa", "aa", "aaa".
//Note:
//The input string length won't exceed 1000.

class PalindromicSubstrings {
    int result = 0;
    public int countSubstrings(String s) {
        if(s == null || s.length() == 0) {
            return 0;
        }
        
        for(int i = 0; i < s.length(); i++) {
            extendPalindrome(s, i, i);
            extendPalindrome(s, i, i + 1);
        }
        
        return result;
    }
    
    public void extendPalindrome(String s, int left, int right) {
        while(left >= 0 && right < s.length() && s.charAt(left) == s.charAt(right)) {
            result++;
            left--;
            right++;
        }
    }
}
public class ValidPalindrome {
    public boolean isPalindrome(String s) {
        int left = 0;
        int right = s.length() - 1;
        
        while(left < right) {
            while(!Character.isLetterOrDigit(s.charAt(left)) && left < right) {
                left++;
            }

            while(!Character.isLetterOrDigit(s.charAt(right)) && right > left) {
                right--;
            }
            
            if(Character.toLowerCase(s.charAt(left)) != Character.toLowerCase(s.charAt(right))) {
                return false;
            }
            
            left++;
            right--;
        }
        
        return true;
    }
}
public class PalindromePermutation {
    public boolean canPermutePalindrome(String s) {
        char[] characters = new char[256];
        
        for(int i = 0; i < s.length(); i++) {
            characters[s.charAt(i)]++;
        }
        
        int oddCount = 0;
        
        for(int i = 0; i < characters.length; i++) {
            if(!(characters[i] % 2 == 0)) {
                oddCount++;
                
                if(oddCount > 1) {
                    return false;
                }
            }
        }
        
        return true;
    }
}
//Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

//Example:
//Input: "babad"
//Output: "bab"

//Note: "aba" is also a valid answer.

//Example:
//Input: "cbbd"
//Output: "bb"

class LongestPalindromicSubstring {
    public String longestPalindrome(String s) {
        if(s == null || s.length() == 0) {
            return "";
        }
        
        String longestPalindromicSubstring = "";
        for(int i = 0; i < s.length(); i++) {
            for(int j = i + 1; j <= s.length(); j++) {
                if(j - i > longestPalindromicSubstring.length() && isPalindrome(s.substring(i, j))) {
                    longestPalindromicSubstring = s.substring(i, j);
                }
            }
        }
        
        return longestPalindromicSubstring;
    }
    
    public boolean isPalindrome(String s) {
        int i = 0;
        int j = s.length() - 1;
        while(i <= j) {
            if(s.charAt(i++) != s.charAt(j--)) {
                return false;
            }
        }
        
        return true;
    }
}
//Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.
//
//For example, given n = 3, a solution set is:
//
//[
  //"((()))",
  //"(()())",
  //"(())()",
  //"()(())",
  //"()()()"
//]

class GenerateParentheses {
    public List<String> generateParenthesis(int n) {
        List<String> result = new ArrayList<String>();
        generateParenthesisRecursive(result, "", 0, 0, n);
        
        return result;
    }
    
    public void generateParenthesisRecursive(List<String> result, String current, int open, int close, int n) {
        if(current.length() == n * 2) {
            result.add(current);
            return;
        }
        
        if(open < n) {
            generateParenthesisRecursive(result, current + "(", open + 1, close, n);
        }
        
        if(close < open) {
            generateParenthesisRecursive(result, current + ")", open, close + 1, n);
        }
    }
}
// Write a function that takes a string as input and reverse only the vowels of a string.

// Example 1:
// Given s = "hello", return "holle".

// Example 2:
// Given s = "leetcode", return "leotcede".

// Note:
// The vowels does not include the letter "y".

public class ReverseVowelsOfAString {
    public String reverseVowels(String s) {
        if(s == null || s.length() == 0) {
            return s;
        }
        
        String vowels = "aeiouAEIOU";
        
        char[] chars = s.toCharArray();
        
        int start = 0;
        int end = s.length() - 1;
        
        while(start < end) {
            while(start < end && !vowels.contains(chars[start] + "")) {
                start++;
            }
            
            while(start < end && !vowels.contains(chars[end] + "")) {
                end--;
            }
            
            char temp = chars[start];
            chars[start] = chars[end];
            chars[end] = temp;
            
            start++;
            end--;
        }
        
        return new String(chars);
    }
}
// Given two strings S and T, determine if they are both one edit distance apart.

public class OneEditDistance {
    public boolean isOneEditDistance(String s, String t) {
        //iterate through the length of the smaller string
        for(int i = 0; i < Math.min(s.length(), t.length()); i++) {
            //if the current characters of the two strings are not equal
            if(s.charAt(i) != t.charAt(i)) {
                //return true if the remainder of the two strings are equal, false otherwise
                if(s.length() == t.length()) {
                    return s.substring(i + 1).equals(t.substring(i + 1));
                } else if(s.length() < t.length()) {
                    //return true if the strings would be the same if you deleted a character from string t
                    return s.substring(i).equals(t.substring(i + 1));
                    
                } else {
                    //return true if the strings would be the same if you deleted a character from string s
                    return t.substring(i).equals(s.substring(i + 1));
                }
            }
        }
        
        //if all characters match for the length of the two strings check if the two strings' lengths do not differ by more than 1
        return Math.abs(s.length() - t.length()) == 1;
    }
}
//Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
//
//Examples:
//
//s = "leetcode"
//return 0.
//
//s = "loveleetcode",
//return 2.
//Note: You may assume the string contain only lowercase letters.

class FirstUniqueCharacterInAString {
    public int firstUniqChar(String s) {
        HashMap<Character, Integer> characters = new HashMap<Character, Integer>();
        for(int i = 0; i < s.length(); i++) {
            char current = s.charAt(i);
            if(characters.containsKey(current)) {
                characters.put(current, -1);
            } else {
                characters.put(current, i);
            }
        }
        
        int min = Integer.MAX_VALUE;
        for(char c: characters.keySet()) {
            if(characters.get(c) > -1 && characters.get(c) < min) {
                min = characters.get(c);
            }
        }
        
        return min == Integer.MAX_VALUE ? -1 : min;
        
    }
}
// The count-and-say sequence is the sequence of integers beginning as follows:
// 1, 11, 21, 1211, 111221, ...

// 1 is read off as "one 1" or 11.
// 11 is read off as "two 1s" or 21.
// 21 is read off as "one 2, then one 1" or 1211.
// Given an integer n, generate the nth sequence.

// Note: The sequence of integers will be represented as a string.

public class CountAndSay {
    public String countAndSay(int n) {
        String s = "1";

        for(int i = 1; i < n; i++) {
            s = helper(s);
        }
        
        return s;
    }
    
    public String helper(String s) {
        StringBuilder sb = new StringBuilder();
        char c = s.charAt(0);
        int count = 1;
        
        for(int i = 1; i < s.length(); i++) {
            if(s.charAt(i) == c) {
                count++;
            } else {
                sb.append(count);
                sb.append(c);
                c = s.charAt(i);
                count = 1;
            }
        }
        
        sb.append(count);
        sb.append(c);

        return sb.toString();
    }
}
// Given a string, find the length of the longest substring T that contains at most k distinct characters.

// For example, Given s = “eceba” and k = 2,

// T is "ece" which its length is 3.

public class LongestSubstringWithAtMostKDistinctCharacters {
    public int lengthOfLongestSubstringKDistinct(String s, int k) {
        
    int[] count = new int[256];     // there are 256 ASCII characters in the world
    
    int i = 0;  // i will be behind j
    int num = 0;
    int res = 0;
    
    for (int j = 0; j < s.length(); j++) {
        if (count[s.charAt(j)] == 0) {    // if count[s.charAt(j)] == 0, we know that it is a distinct character
            num++;
        }
        
        count[s.charAt(j)]++;

        while (num > k && i < s.length()) {     // sliding window
            count[s.charAt(i)]--;

            if (count[s.charAt(i)] == 0){ 
                num--;
            }

            i++;
        }

        res = Math.max(res, j - i + 1);
    }

    return res;
    }
}
// Given two non-negative integers num1 and num2 represented as strings, return the product of num1 and num2.

// Note:

    // The length of both num1 and num2 is < 110.
    // Both num1 and num2 contains only digits 0-9.
    // Both num1 and num2 does not contain any leading zero.
    // You must not use any built-in BigInteger library or convert the inputs to integer directly.

public class MultiplyStrings {
    public String multiply(String num1, String num2) {
        int m = num1.length();
        int n = num2.length();
        int[] pos = new int[m + n];
        
        for(int i = m - 1; i >= 0; i--) {
            for(int j = n - 1; j >= 0; j--) {
                int mul = (num1.charAt(i) - '0') * (num2.charAt(j) - '0');
                int p1 = i + j;
                int p2 = i + j + 1;
                int sum = mul + pos[p2];
                
                pos[p1] += sum / 10;
                pos[p2] = (sum) % 10;
            }
        }
        
        StringBuilder sb = new StringBuilder();

        for(int p : pos) if(!(sb.length() == 0 && p == 0)) {
            sb.append(p);
        }
        
        return sb.length() == 0 ? "0" : sb.toString();
    }
}
// Convert a non-negative integer to its english words representation. Given input is guaranteed to be less than 231 - 1.

// For example,

// 123 -> "One Hundred Twenty Three"
// 12345 -> "Twelve Thousand Three Hundred Forty Five"
// 1234567 -> "One Million Two Hundred Thirty Four Thousand Five Hundred Sixty Seven"

public class IntegerToEnglishWords {
    private final String[] LESS_THAN_20 = { "", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven", "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen" };
    private final String[] TENS = { "", "Ten", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety" };
    private final String[] THOUSANDS = { "", "Thousand", "Million", "Billion" };
    
    public String numberToWords(int num) {
        if(num == 0) {
            return "Zero";
        }
        
        int i = 0;
        String words = "";
        
        while(num > 0) {
            if(num % 1000 != 0) {
                words = helper(num % 1000) + THOUSANDS[i] + " " + words;
            }
            
            num /= 1000;
            i++;
        }
        
        return words.trim();
    }
    
    private String helper(int num) {
        if(num == 0) {
            return "";
        } else if(num < 20) {
            return LESS_THAN_20[num] + " ";
        } else if(num < 100) {
            return TENS[num / 10] + " " + helper(num % 10);
        } else {
            return LESS_THAN_20[num / 100] + " Hundred " + helper(num % 100);
        }
    }
}
// Given two words word1 and word2, find the minimum number of steps required to convert word1 to word2. (each operation is counted as 1 step.)

// You have the following 3 operations permitted on a word:

// a) Insert a character
// b) Delete a character
// c) Replace a character

public class EditDistance {
    public int minDistance(String word1, String word2) {
        int m = word1.length();
        int n = word2.length();
        
        int[][] dp = new int[m + 1][n + 1];

        for(int i = 0; i <= m; i++) {
            dp[i][0] = i;
        }
        
        for(int i = 0; i <= n; i++) {
            dp[0][i] = i;
        }
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(word1.charAt(i) == word2.charAt(j)) {
                    dp[i + 1][j + 1] = dp[i][j];
                } else {
                    int a = dp[i][j];
                    int b = dp[i][j + 1];
                    int c = dp[i + 1][j];
                    
                    dp[i + 1][j + 1] = Math.min(a, Math.min(b, c));
                    dp[i + 1][j + 1]++;
                }
            }
        }
        
        return dp[m][n];
    }
}
public class LongestPalindrome {
    public int longestPalindrome(String s) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        
        int count = 0;
        
        for(int i = 0; i < s.length(); i++) {
            if(!map.containsKey(s.charAt(i))) {
                map.put(s.charAt(i), (int)(s.charAt(i)));
            } else {
                map.remove(s.charAt(i));
                count++;
            }
        }
        
        return map.isEmpty() ? count * 2 : count * 2 + 1;
    }
}
//Given an input string, reverse the string word by word.
//For example,
//Given s = "the sky is blue",
//return "blue is sky the".

public class ReverseWordsInAString {
    public String reverseWords(String s) {
        String[] words = s.trim().split("\\s+");
        String result = "";
        for(int i = words.length - 1; i > 0; i--) {
            result += words[i] + " ";
        }
        
        return result + words[0];
    }
}
// Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

// The brackets must close in the correct order, "()" and "()[]{}" are all valid but "(]" and "([)]" are not.

public class ValidParentheses {
    public boolean isValid(String s) {
        if(s.length() % 2 == 1) {
            return false;
        }
        
        Stack<Character> stack = new Stack<Character>();
        
        for(int i = 0; i < s.length(); i++) {
            if(s.charAt(i) == '(' || s.charAt(i) == '[' || s.charAt(i) == '{') {
                stack.push(s.charAt(i));
            } else if(s.charAt(i) == ')' && !stack.isEmpty() && stack.peek() == ')') {
                stack.pop();
            } else if(s.charAt(i) == ']' && !stack.isEmpty() && stack.peek() == ']') {
                stack.pop();
            } else if(s.charAt(i) == '}' && !stack.isEmpty() && stack.peek() == '}') {
                stack.pop();
            } else {
                return false;
            }
        }
        
        return stack.isEmpty();
    }
}
class LongestCommonPrefix {
    public String longestCommonPrefix(String[] strs) {
        if(strs == null || strs.length == 0) {
            return "";
        }
        
        String s = strs[0];
        for(int i = 0; i < s.length(); i++) {
            char current = s.charAt(i);
            for(int j = 1; j < strs.length; j++) {
                if(i >= strs[j].length() || strs[j].charAt(i) != current) {
                    return s.substring(0, i);
                }
            }
        }
        
        return s;
    }
}
// Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

// For example,
// S = "ADOBECODEBANC"
// T = "ABC"
// Minimum window is "BANC".

// Note:
    // If there is no such window in S that covers all characters in T, return the empty string "".
    // If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.

public class MinimumWindowSubstring {
    public String minWindow(String s, String t) {
        HashMap<Character, Integer> map = new HashMap<>();
        
        for(char c : s.toCharArray()) {
            map.put(c, 0);
        }
        
        for(char c : t.toCharArray()) {
            if(map.containsKey(c)) {
                map.put(c, map.get(c)+ 1);
            } else {
                return "";
            }
        }
        
        int start = 0;
        int end = 0;
        int minStart = 0;
        int minLength = Integer.MAX_VALUE;
        int counter = t.length();
        
        while(end < s.length()) {
            char c1 = s.charAt(end);
            
            if(map.get(c1) > 0) {
                counter--;
            }
            
            map.put(c1, map.get(c1) - 1);
            end++;
            
            while(counter == 0) {
                if(minLength > end - start) {
                    minLength = end - start;
                    minStart = start;
                }
                
                char c2 = s.charAt(start);
                map.put(c2, map.get(c2) + 1);
                
                if(map.get(c2) > 0) {
                    counter++;
                }
                
                start++;
            }
        }
        
        return minLength == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLength);
    }
}
// Given a roman numeral, convert it to an integer.

// Input is guaranteed to be within the range from 1 to 3999

public class RomanToInteger {
    public int romanToInt(String s) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        
        map.put('I', 1);
        map.put('V', 5);
        map.put('X', 10);
        map.put('L', 50);
        map.put('C', 100);
        map.put('D', 500);
        map.put('M', 1000);
        
        int total = 0;

        for(int i = 0; i < s.length() - 1; i++) {
            if(map.get(s.charAt(i)) < map.get(s.charAt(i + 1))) {
                total -= map.get(s.charAt(i));
            } else {
                total += map.get(s.charAt(i));
            }
        }
        
        total += map.get(s.charAt(s.length() - 1));
        
        return total;
    }
}
//Initially, there is a Robot at position (0, 0). Given a sequence of its moves, judge if this robot makes a circle, which means it moves back to the original place.
//
//The move sequence is represented by a string. And each move is represent by a character. The valid robot moves are R (Right), L (Left), U (Up) and D (down). The output should be true or false representing whether the robot makes a circle.
//
//Example 1:
//Input: "UD"
//Output: true
//Example 2:
//Input: "LL"
//Output: false

class JudgeRouteCircle {
    public boolean judgeCircle(String moves) {
        int UD = 0;
        int LR = 0;
        for(int i = 0; i < moves.length(); i++) {
            if(moves.charAt(i) == 'U') {
                UD++;
            } else if(moves.charAt(i) == 'D') {
                UD--;
            } else if(moves.charAt(i) == 'L') {
                LR++;
            } else if(moves.charAt(i) == 'R') {
                LR--;
            }
        }
        
        return UD == 0 && LR == 0;
    }
}
// Given two binary strings, return their sum (also a binary string).

// For example,
// a = "11"
// b = "1"
// Return "100"

public class AddBinary {
    public String addBinary(String a, String b) {
        StringBuilder result = new StringBuilder();
        
        int carry = 0;
        int i = a.length() - 1;
        int j = b.length() - 1;
        
        while(i >= 0 || j >= 0) {
            int sum = carry;

            if(i >= 0) {
                sum += a.charAt(i--) - '0';
            }

            if(j >= 0) {
                sum += b.charAt(j--) - '0';
            }

            result.append(sum % 2);
            carry = sum / 2;
        }

        if(carry != 0) {
            result.append(carry);
        }

        return result.reverse().toString();
    }
}
// A message containing letters from A-Z is being encoded to numbers using the following mapping:

// 'A' -> 1
// 'B' -> 2
// ...
// 'Z' -> 26

// Given an encoded message containing digits, determine the total number of ways to decode it.

// For example,
// Given encoded message "12", it could be decoded as "AB" (1 2) or "L" (12).

// The number of ways decoding "12" is 2.

public class DecodeWays {
    public int numDecodings(String s) {
        int n = s.length();

        if(n == 0) {
            return 0;
        }
        
        int[] dp = new int[n + 1];
        dp[n] = 1;
        dp[n - 1] = s.charAt(n - 1) != '0' ? 1 : 0;
        
        for(int i = n - 2; i >= 0; i--) {
            if(s.charAt(i) == '0') {
                continue;
            } else {
                dp[i] = (Integer.parseInt(s.substring(i, i + 2)) <= 26) ? dp[i + 1] + dp[i + 2] : dp[i + 1];
            }
        }
        
        return dp[0];
    }
}
//Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.
//
//For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].
//
//Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

class DailyTemperatures {
    public int[] dailyTemperatures(int[] temperatures) {
        int[] result = new int[temperatures.length];
        Stack<Integer> stack = new Stack<Integer>();
        for(int i = 0; i < temperatures.length; i++) {
            while(!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int index = stack.pop();
                result[index] = i - index;
            }
            stack.push(i);
        }
        
        return result;
    }
}
//Given the running logs of n functions that are executed in a nonpreemptive single threaded CPU, find the exclusive time of these functions.

//Each function has a unique id, start from 0 to n-1. A function may be called recursively or by another function.

//A log is a string has this format : function_id:start_or_end:timestamp. For example, "0:start:0" means function 0 starts from the very beginning of time 0. "0:end:0" means function 0 ends to the very end of time 0.

//Exclusive time of a function is defined as the time spent within this function, the time spent by calling other functions should not be considered as this function's exclusive time. You should return the exclusive time of each function sorted by their function id.

//Example 1:
//Input:
//n = 2
//logs = 
//["0:start:0",
 //"1:start:2",
 //"1:end:5",
 //"0:end:6"]
//Output:[3, 4]
//Explanation:
//Function 0 starts at time 0, then it executes 2 units of time and reaches the end of time 1. 
//Now function 0 calls function 1, function 1 starts at time 2, executes 4 units of time and end at time 5.
//Function 0 is running again at time 6, and also end at the time 6, thus executes 1 unit of time. 
//So function 0 totally execute 2 + 1 = 3 units of time, and function 1 totally execute 4 units of time.
//Note:
//Input logs will be sorted by timestamp, NOT log id.
//Your output should be sorted by function id, which means the 0th element of your output corresponds to the exclusive time of function 0.
//Two functions won't start or end at the same time.
//Functions could be called recursively, and will always end.
//1 <= n <= 100

class ExclusiveTimeOfFunctions {
    public int[] exclusiveTime(int n, List<String> logs) {
        Stack<Integer> stack = new Stack <Integer>();
        int[] result = new int[n];
        String[] current = logs.get(0).split(":");
        stack.push(Integer.parseInt(current[0]));
        int i = 1;
        int previous = Integer.parseInt(current[2]);
        while (i < logs.size()) {
            current = logs.get(i).split(":");
            if (current[1].equals("start")) {
                if (!stack.isEmpty()) {
                    result[stack.peek()] += Integer.parseInt(current[2]) - previous;
                }
                stack.push(Integer.parseInt(current[0]));
                previous = Integer.parseInt(current[2]);
            } else {
                result[stack.peek()] += Integer.parseInt(current[2]) - previous + 1;
                stack.pop();
                previous = Integer.parseInt(current[2]) + 1;
            }
            i++;
        }
        return result;
    }
}

// Given a nested list of integers, implement an iterator to flatten it.

// Each element is either an integer, or a list -- whose elements may also be integers or other lists.

// Example 1:
// Given the list [[1,1],2,[1,1]],

// By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,1,2,1,1].

// Example 2:
// Given the list [1,[4,[6]]],

// By calling next repeatedly until hasNext returns false, the order of elements returned by next should be: [1,4,6].

/**
 * // This is the interface that allows for creating nested lists.
 * // You should not implement it, or speculate about its implementation
 * public interface NestedInteger {
 *
 *     // @return true if this NestedInteger holds a single integer, rather than a nested list.
 *     public boolean isInteger();
 *
 *     // @return the single integer that this NestedInteger holds, if it holds a single integer
 *     // Return null if this NestedInteger holds a nested list
 *     public Integer getInteger();
 *
 *     // @return the nested list that this NestedInteger holds, if it holds a nested list
 *     // Return null if this NestedInteger holds a single integer
 *     public List<NestedInteger> getList();
 * }
 */
public class FlattenNestedListIterator implements Iterator<Integer> {
    Stack<NestedInteger> stack = new Stack<NestedInteger>();

    public NestedIterator(List<NestedInteger> nestedList) {
        for(int i = nestedList.size() - 1; i >= 0; i--) {
            stack.push(nestedList.get(i));
        }
    }

    @Override
    public Integer next() {
        return stack.pop().getInteger();
    }

    @Override
    public boolean hasNext() {
        while(!stack.isEmpty()) {
            NestedInteger current = stack.peek();

            if(current.isInteger()) {
                return true;
            }

            stack.pop();

            for(int i = current.getList().size() - 1;  i >= 0; i--) {
                stack.push(current.getList().get(i));
            }
        }
        
        return false;
    }
}

/**
 * Your NestedIterator object will be instantiated and called as such:
 * NestedIterator i = new NestedIterator(nestedList);
 * while (i.hasNext()) v[f()] = i.next();
 */
//Design a stack that supports push, pop, top, and retrieving the minimum element in constant time.
//push(x) -- Push element x onto stack.
//pop() -- Removes the element on top of the stack.
//top() -- Get the top element.
//getMin() -- Retrieve the minimum element in the stack.

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack obj = new MinStack();
 * obj.push(x);
 * obj.pop();
 * int param_3 = obj.top();
 * int param_4 = obj.getMin();
 */
class MinStack {
    class Node {
        int data;
        int min;
        Node next;
        
        public Node(int data, int min) {
            this.data = data;
            this.min = min;
            this.next = null;
        }
    }
    Node head;
    
    /** initialize your data structure here. */
    public MinStack() {
        
    }
    
    public void push(int x) {
        if(head == null) {
            head = new Node(x, x);
        } else {
            Node newNode = new Node(x, Math.min(x, head.min));
            newNode.next = head;
            head = newNode;
        }
    }
    
    public void pop() {
        head = head.next;
    }
    
    public int top() {
        return head.data;
    }
    
    public int getMin() {
        return head.min;
    }
}

// Given n non-negative integers representing an elevation map where the width of each bar is 1, compute how much water it is able to trap after raining.

// For example, 
// Given [0,1,0,2,1,0,1,3,2,1,2,1], return 6.

public class TrappingRainWater {
    public int trap(int[] height) {
        int water = 0;
        
        int leftIndex = 0;
        int rightIndex = height.length - 1;
        
        int leftMax = 0;
        int rightMax = 0;
        
        while(leftIndex <= rightIndex) {
            leftMax = Math.max(leftMax, height[leftIndex]);
            rightMax = Math.max(rightMax, height[rightIndex]);
            
            if(leftMax < rightMax) {
                water += leftMax - height[leftIndex];
                leftIndex++;
            } else {
                water += rightMax - height[rightIndex];
                rightIndex--;
            }
        }
        
        return water;
    }
}

// Given an encoded string, return it's decoded string.

// The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly k times. Note that k is guaranteed to be a positive integer.

// You may assume that the input string is always valid; No extra white spaces, square brackets are well-formed, etc.

// Furthermore, you may assume that the original data does not contain any digits and that digits are only for those repeat numbers, k. For example, there won't be input like 3a or 2[4].

public class DecodeString {
    public String decodeString(String s) {
        //declare empty string
        String decoded = "";
        
        //initialize stack to hold counts
        Stack<Integer> countStack = new Stack<Integer>();
        
        //initalize stack to hold decoded string
        Stack<String> decodedStack = new Stack<String>();
        
        //initialize index to zero
        int index = 0;
        
        //iterate through entire string
        while(index < s.length()) {
            //if the current character is numeric...
            if(Character.isDigit(s.charAt(index))) {
                int count = 0;
                
                //determine the number
                while(Character.isDigit(s.charAt(index))) {
                    count = 10 * count + (s.charAt(index) - '0');
                    index++;
                }
                
                //push the number onto the count stack
                countStack.push(count);
            } else if(s.charAt(index) == '[') {
                //if the current character is an opening bracket
                decodedStack.push(decoded);
                decoded = "";
                index++;
            } else if(s.charAt(index) == ']') {
                //if the current character is a closing bracket
                StringBuilder temp = new StringBuilder(decodedStack.pop());
                int repeatTimes = countStack.pop();
                
                for(int i = 0; i < repeatTimes; i++) {
                    temp.append(decoded);
                }
                
                decoded = temp.toString();
                index++;
            } else {
                //otherwise, append the current character to the decoded string
                decoded += s.charAt(index);
                index++;
            }
        }
        
        //return the decoded string
        return decoded;
    }
}
// Implement an iterator over a binary search tree (BST). Your iterator will be initialized with the root node of a BST.

// Calling next() will return the next smallest number in the BST.

// Note: next() and hasNext() should run in average O(1) time and uses O(h) memory, where h is the height of the tree.

/**
 * Definition for binary tree
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */

public class BinarySearchTreeIterator {
    Stack<TreeNode> stack;

    public BSTIterator(TreeNode root) {
        stack = new Stack<TreeNode>();
        
        while(root != null) {
            stack.push(root);
            root = root.left;
        }
    }

    /** @return whether we have a next smallest number */
    public boolean hasNext() {
        return stack.isEmpty() ? false : true;
    }

    /** @return the next smallest number */
    public int next() {
        TreeNode nextSmallest = stack.pop();
        TreeNode addToStack = nextSmallest.right;
        
        while(addToStack != null) {
            stack.add(addToStack);
            addToStack = addToStack.left;
        }
        
        return nextSmallest.val;
    }
}

/**
 * Your BSTIterator will be called like this:
 * BSTIterator i = new BSTIterator(root);
 * while (i.hasNext()) v[f()] = i.next();
 */
// A character in UTF8 can be from 1 to 4 bytes long, subjected to the following rules:

// For 1-byte character, the first bit is a 0, followed by its unicode code.
// For n-bytes character, the first n-bits are all one's, the n+1 bit is 0, followed by n-1 bytes with most significant 2 bits being 10.
// This is how the UTF-8 encoding would work:

//    Char. number range  |        UTF-8 octet sequence
//       (hexadecimal)    |              (binary)
//    --------------------+---------------------------------------------
//    0000 0000-0000 007F | 0xxxxxxx
//    0000 0080-0000 07FF | 110xxxxx 10xxxxxx
//    0000 0800-0000 FFFF | 1110xxxx 10xxxxxx 10xxxxxx
//    0001 0000-0010 FFFF | 11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
// Given an array of integers representing the data, return whether it is a valid utf-8 encoding.

// Note:
// The input is an array of integers. Only the least significant 8 bits of each integer is used to store the data. This means each integer represents only 1 byte of data.

// Example 1:

// data = [197, 130, 1], which represents the octet sequence: 11000101 10000010 00000001.

// Return true.
// It is a valid utf-8 encoding for a 2-bytes character followed by a 1-byte character.
// Example 2:

// data = [235, 140, 4], which represented the octet sequence: 11101011 10001100 00000100.

// Return false.
// The first 3 bits are all one's and the 4th bit is 0 means it is a 3-bytes character.
// The next byte is a continuation byte which starts with 10 and that's correct.
// But the second continuation byte does not start with 10, so it is invalid.

public class Utf8Validation {
    public boolean validUtf8(int[] data) {
        int count = 0;

        for(int i : data) {
            if(count == 0) {
                if((i >> 5) == 0b110) {
                    count = 1;
                } else if((i >> 4) == 0b1110) {
                    count = 2;
                } else if((i >> 3) == 0b11110) {
                    count = 3;
                } else if((i >> 7) == 0b1) {
                    return false;
                }
            } else {
                if((i >> 6) != 0b10) {
                    return false;
                }

                count--;
            }
        }
        
        return count == 0;
    }
}
//Given an integer, write a function to determine if it is a power of two.
//
//Example 1:
//
//Input: 1
//Output: true
//Example 2:
//
//Input: 16
//Output: true
//Example 3:
//
//Input: 218
//Output: false

class PowerOfTwo {
    public boolean isPowerOfTwo(int n) {
        long i = 1;
        while(i < n) {
            i <<= 1;
        }
        
        return i == n;
    }
}
// A binary watch has 4 LEDs on the top which represent the hours (0-11), and the 6 LEDs on the bottom represent the minutes (0-59).

// Each LED represents a zero or one, with the least significant bit on the right.

// For example, the above binary watch reads "3:25".

// Given a non-negative integer n which represents the number of LEDs that are currently on, return all possible times the watch could represent.

// Example:

// Input: n = 1
// Return: ["1:00", "2:00", "4:00", "8:00", "0:01", "0:02", "0:04", "0:08", "0:16", "0:32"]
// Note:
// The order of output does not matter.
// The hour must not contain a leading zero, for example "01:00" is not valid, it should be "1:00".
// The minute must be consist of two digits and may contain a leading zero, for example "10:2" is not valid, it should be "10:02".

public class BinaryWatch {
    public List<String> readBinaryWatch(int num) {
        ArrayList<String> allTimes = new ArrayList<String>();
        
        //iterate through all possible time combinations
        for(int i = 0; i < 12; i++) {
            for(int j = 0; j < 60; j++) {
                //if the current number and n have the same number of bits the time is possible
                if(Integer.bitCount(i * 64 + j) == num) {
                    //add the current time to all times arraylist
                    allTimes.add(String.format("%d:%02d", i, j));
                }
            }
        }
        
        return allTimes;
    }
}
// Write a function that takes an unsigned integer and returns the number of ’1' bits it has (also known as the Hamming weight).

// For example, the 32-bit integer ’11' has binary representation 00000000000000000000000000001011, so the function should return 3.

public class NumberOfOneBits {
    // you need to treat n as an unsigned value
    public int hammingWeight(int n) {
        if(n == 0) {
            return 0;
        }
        
        int count = 0;
        
        while(n != 0) {
            count += (n) & 1;
            n >>>= 1;
        }
        
        return count;
    }
}
// Given a non negative integer number num. For every numbers i in the range 0 ≤ i ≤ num calculate the number of 1's in their binary representation and return them as an array.

// Example:
// For num = 5 you should return [0,1,1,2,1,2].

public class CountingBits {
    public int[] countBits(int num) {
        int[] bits = new int[num + 1];
        
        bits[0] = 0;
        
        for(int i = 1; i <= num; i++) {
            bits[i] = bits[i >> 1] + (i & 1);
        }
        
        return bits;
    }
}
// Calculate the sum of two integers a and b, but you are not allowed to use the operator + and -.

// Example:
// Given a = 1 and b = 2, return 3.

public class SumOfTwoIntegers {
    public int getSum(int a, int b) {
        if(a == 0) {
            return b;
        }

        if(b == 0) {
            return a;
        }
        
        while(b != 0) {
            int carry = a & b;
            a = a ^ b;
            b = carry << 1;
        }
        
        return a;
    }
}
// Given a string array words, find the maximum value of length(word[i]) * length(word[j]) where the two words do not share common letters. You may assume that each word will contain only lower case letters. If no such two words exist, return 0.

// Example 1:
// Given ["abcw", "baz", "foo", "bar", "xtfn", "abcdef"]
// Return 16
// The two words can be "abcw", "xtfn".

// Example 2:
// Given ["a", "ab", "abc", "d", "cd", "bcd", "abcd"]
// Return 4
// The two words can be "ab", "cd".

// Example 3:
// Given ["a", "aa", "aaa", "aaaa"]
// Return 0
// No such pair of words.

public class MaximumProductOfWordLengths {
    public int maxProduct(String[] words) {
        if(words.length == 0 || words == null) {
            return 0;
        }
        
        int length = words.length;
        int[] value = new int[length];
        int max = 0;
        
        for(int i = 0; i < length; i++) {
            String temp = words[i];
            
            value[i] = 0;
            
            for(int j = 0; j < temp.length(); j++) {
                value[i] |= 1 << (temp.charAt(j) - 'a');
            }
        }
        
        
        for(int i = 0; i < length; i++) {
            for(int j = 1; j < length; j++) {
                if((value[i] & value[j]) == 0 && (words[i].length() * words[j].length()) > max) {
                    max = words[i].length() * words[j].length();
                }
            }
        }
        
        return max;
    }
}
// The Hamming distance between two integers is the number of positions at which the corresponding bits are different.

// Given two integers x and y, calculate the Hamming distance.

// Note:
// 0 ≤ x, y < 2^31.

// Example:

// Input: x = 1, y = 4

// Output: 2

// Explanation:
// 1   (0 0 0 1)
// 4   (0 1 0 0)
//        ↑   ↑

// The above arrows point to positions where the corresponding bits are different.

public class HammingDistance {
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }
}
class ValidAnagram {
    public boolean isAnagram(String s, String t) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        for(char c: s.toCharArray()) {
            if(map.containsKey(c)) {
                map.put(c, map.get(c) + 1);
            }
            else {
                map.put(c, 1);
            }
        }
        
        for(char c: t.toCharArray()) {
            if(map.containsKey(c)) {
                map.put(c, map.get(c) - 1);
            }
            else {
                return false;
            }
        }
        
        for(char c: map.keySet()) {
            if(map.get(c) != 0) {
                return false;
            }
        }
        
        return true;
    }
}
// An abbreviation of a word follows the form <first letter><number><last letter>. Below are some examples of word abbreviations:

// a) it                      --> it    (no abbreviation)

//      1
// b) d|o|g                   --> d1g

//               1    1  1
//      1---5----0----5--8
// c) i|nternationalizatio|n  --> i18n

//               1
//      1---5----0
// d) l|ocalizatio|n          --> l10n
// Assume you have a dictionary and given a word, find whether its abbreviation is unique in the dictionary. A word's abbreviation is unique if no other word from the dictionary has the same abbreviation.

// Example: 
// Given dictionary = [ "deer", "door", "cake", "card" ]

// isUnique("dear") -> 
// false

// isUnique("cart") -> 
// true

// isUnique("cane") -> 
// false

// isUnique("make") -> 
// true

import java.util.ArrayList;

public class UniqueWordAbbreviation {
    HashMap<String, String> map;

    public ValidWordAbbr(String[] dictionary) {
        this.map = new HashMap<String, String>();
        
        for(String word : dictionary) {
            String key = getKey(word);
            
            if(map.containsKey(key)) {
                if(!map.get(key).equals(word)) {
                    map.put(key, "");
                }
            } else {
                map.put(key, word);
            }
        }
    }

    public boolean isUnique(String word) {
        return !map.containsKey(getKey(word))||map.get(getKey(word)).equals(word);
    }
    
    public String getKey(String word) {
        if(word.length() <= 2) {
            return word;
        }
        
        return word.charAt(0) + Integer.toString(word.length() - 2) + word.charAt(word.length() - 1);
    }
}


// Your ValidWordAbbr object will be instantiated and called as such:
// ValidWordAbbr vwa = new ValidWordAbbr(dictionary);
// vwa.isUnique("Word");
// vwa.isUnique("anotherWord");
//Given an array of integers and an integer k, find out whether there are two distinct indices i and 
//j in the array such that nums[i] = nums[j] and the absolute difference between i and j is at most k.

class ContainsDuplicatesII {
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i = 0; i < nums.length; i++) {
            int current = nums[i];
            if(map.containsKey(current) && i - map.get(current) <= k) {
                return true;
            } else {
                map.put(current, i);
            }
        }
        
        return false;
    }
}

//TinyURL is a URL shortening service where you enter a URL such as https://leetcode.com/problems/design-tinyurl 
//and it returns a short URL such as http://tinyurl.com/4e9iAk.
//
//Design the encode and decode methods for the TinyURL service. There is no restriction on how your 
//encode/decode algorithm should work. You just need to ensure that a URL can be encoded to a tiny URL 
//and the tiny URL can be decoded to the original URL.

public class EncodeAndDecodeTinyURL {
    HashMap<String, String> map = new HashMap<String, String>();
    String characters = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    int count = 1;

    public String getKey() {
        String key = "";
        while(count > 0) {
            count--;
            key += characters.charAt(count);
            count /= characters.length();
        }
        
        return key;
    }
    
    // Encodes a URL to a shortened URL.
    public String encode(String longUrl) {
        String key = getKey();
        map.put(key, longUrl);
        count++;
            
        return "http://tinyurl.com/" + key;
    }

    // Decodes a shortened URL to its original URL.
    public String decode(String shortUrl) {
        return map.get(shortUrl.replace("http://tinyurl.com/", ""));
    }
}

// Your Codec object will be instantiated and called as such:
// Codec codec = new Codec();
// codec.decode(codec.encode(url));
//Given a list of daily temperatures, produce a list that, for each day in the input, tells you how many days you would have to wait until a warmer temperature. If there is no future day for which this is possible, put 0 instead.
//
//For example, given the list temperatures = [73, 74, 75, 71, 69, 72, 76, 73], your output should be [1, 1, 4, 2, 1, 1, 0, 0].
//
//Note: The length of temperatures will be in the range [1, 30000]. Each temperature will be an integer in the range [30, 100].

class DailyTemperatures {
    public int[] dailyTemperatures(int[] temperatures) {
        int[] result = new int[temperatures.length];
        Stack<Integer> stack = new Stack<Integer>();
        for(int i = 0; i < temperatures.length; i++) {
            while(!stack.isEmpty() && temperatures[i] > temperatures[stack.peek()]) {
                int index = stack.pop();
                result[index] = i - index;
            }
            stack.push(i);
        }
        
        return result;
    }
}
//Design a data structure that supports all following operations in average O(1) time.

//insert(val): Inserts an item val to the set if not already present.
//remove(val): Removes an item val from the set if present.
//getRandom: Returns a random element from current set of elements. Each element must have the same probability of being returned.

//Example:
// Init an empty set.
//RandomizedSet randomSet = new RandomizedSet();

// Inserts 1 to the set. Returns true as 1 was inserted successfully.
//randomSet.insert(1);

// Returns false as 2 does not exist in the set.
//randomSet.remove(2);

// Inserts 2 to the set, returns true. Set now contains [1,2].
//randomSet.insert(2);

// getRandom should return either 1 or 2 randomly.
//randomSet.getRandom();

// Removes 1 from the set, returns true. Set now contains [2].
//randomSet.remove(1);

// 2 was already in the set, so return false.
//randomSet.insert(2);

// Since 2 is the only number in the set, getRandom always return 2.
//randomSet.getRandom();

class RandomizedSet {
    HashMap<Integer, Integer> map;
    ArrayList<Integer> values;

    /** Initialize your data structure here. */
    public RandomizedSet() {
        map = new HashMap<Integer, Integer>();
        values = new ArrayList<Integer>();
    }
    
    /** Inserts a value to the set. Returns true if the set did not already contain the specified element. */
    public boolean insert(int val) {
        if(!map.containsKey(val)) {
            map.put(val, val);
            values.add(val);
            return true;
        }
        else {
            return false;
        }
    }
    
    /** Removes a value from the set. Returns true if the set contained the specified element. */
    public boolean remove(int val) {
        if(map.containsKey(val)) {
            map.remove(val);
            values.remove(values.indexOf(val));
            return true;
        }
        return false;
    }
    
    /** Get a random element from the set. */
    public int getRandom() {
        int random = (int)(Math.random() * values.size());
        int valueToReturn = values.get(random);
        return map.get(valueToReturn);
    }
}

/**
 * Your RandomizedSet object will be instantiated and called as such:
 * RandomizedSet obj = new RandomizedSet();
 * boolean param_1 = obj.insert(val);
 * boolean param_2 = obj.remove(val);
 * int param_3 = obj.getRandom();
 */

// A strobogrammatic number is a number that looks the same when rotated 180 degrees (looked at upside down).

// Write a function to determine if a number is strobogrammatic. The number is represented as a string.

// For example, the numbers "69", "88", and "818" are all strobogrammatic.

public class StrobogrammaticNumber {
    public boolean isStrobogrammatic(String num) {
        for(int i = 0, j = num.length() - 1; i <= j; i++, j--) {
            if(!"00 11 88 696".contains(num.charAt(i) + "" + num.charAt(j))) {
                return false;
            }
        }
        
        return true;
    }
}
//Given a string, find the first non-repeating character in it and return it's index. If it doesn't exist, return -1.
//
//Examples:
//
//s = "leetcode"
//return 0.
//
//s = "loveleetcode",
//return 2.
//Note: You may assume the string contain only lowercase letters.

class FirstUniqueCharacterInAString {
    public int firstUniqChar(String s) {
        HashMap<Character, Integer> characters = new HashMap<Character, Integer>();
        for(int i = 0; i < s.length(); i++) {
            char current = s.charAt(i);
            if(characters.containsKey(current)) {
                characters.put(current, -1);
            } else {
                characters.put(current, i);
            }
        }
        
        int min = Integer.MAX_VALUE;
        for(char c: characters.keySet()) {
            if(characters.get(c) > -1 && characters.get(c) < min) {
                min = characters.get(c);
            }
        }
        
        return min == Integer.MAX_VALUE ? -1 : min;
        
    }
}
//Given an array of integers, find if the array contains any duplicates. Your function should return 
//true if any value appears at least twice in the array, and it should return false if every element is distinct.

class ContainsDuplicate {
    public boolean containsDuplicate(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i: nums) {
            if(map.containsKey(i)) {
                return true;
            } else {
                map.put(i, 1);
            }
        }
        
        return false;
    }
}
// You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water. Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells). The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

// Example:

// [[0,1,0,0],
//  [1,1,1,0],
//  [0,1,0,0],
//  [1,1,0,0]]

// Answer: 16

class IslandPerimeter {
    public int islandPerimeter(int[][] grid) {
        int perimeter = 0;
        if(grid == null || grid.length == 0) {
            return perimeter;
        }

        for(int i = 0; i < grid.length; i++) {
            for(int j = 0; j < grid[i].length; j++) {
                if(grid[i][j] == 1) {
                    perimeter += numNeighbors(grid, i, j);
                    return perimeter;
                }
            }
        }

        return perimeter;
    }

    public int numNeighbors(int[][] grid, int x, int y) {
        if(x < 0 || x >= grid.length || y < 0 || y >= grid[x].length || grid[x][y] == 0) {
            return 1;
        }

        if(grid[x][y] == -1) {
            return 0;
        }

        grid[x][y] = -1;
        return numNeighbors(grid, x + 1, y) + 
            numNeighbors(grid, x - 1, y) + 
            numNeighbors(grid, x, y + 1) + 
            numNeighbors(grid, x, y - 1);
    }
}
//Given an array of integers, every element appears three times except for one, 
//which appears exactly once. Find that single one.

//Note:
//Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

class SingleNumberII {
    public int singleNumber(int[] nums) {
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        for(int i: nums) {
            if(map.containsKey(i)) {
                map.put(i, map.get(i) + 1);
            } else {
                map.put(i, 1);
            }
        }
        
        for(int key: map.keySet()) {
            if(map.get(key) == 1) {
                return key;
            }
        }
        
        //no unique integer in nums
        return -1;
    }
}
// Given an array nums and a target value k, find the maximum length of a subarray that sums to k. If there isn't one, return 0 instead.

// Note:
// The sum of the entire nums array is guaranteed to fit within the 32-bit signed integer range.

// Example 1:
// Given nums = [1, -1, 5, -2, 3], k = 3,
// return 4. (because the subarray [1, -1, 5, -2] sums to 3 and is the longest)

// Example 2:
// Given nums = [-2, -1, 2, 1], k = 1,
// return 2. (because the subarray [-1, 2] sums to 1 and is the longest)

// Follow Up:
// Can you do it in O(n) time?

public class MaximumSizeSubarraySumEqualsK {
    public int maxSubArrayLen(int[] nums, int k) {
        if(nums.length == 0) {
            return 0;
        }
        
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        
        int maxLength = 0;
        
        int total = 0;
        
        map.put(0, -1);
        
        for(int i = 0; i < nums.length; i++) {
            total += nums[i];
            if(map.containsKey(total - k)) {
                maxLength = Math.max(maxLength, i - map.get(total - k));
            }

            if(!map.containsKey(total)) {
                map.put(total, i);
            }
        }
        
        return maxLength;
    }
}
// Given two strings s and t which consist of only lowercase letters.

// String t is generated by random shuffling string s and then add one more letter at a random position.

// Find the letter that was added in t.

// Example:

// Input:
// s = "abcd"
// t = "abcde"

// Output:
// e

// Explanation:
// 'e' is the letter that was added.

public class FindTheDifference {
    public char findTheDifference(String s, String t) {
        int charCodeS = 0;
        int charCodeT = 0;
        
        for(int i = 0; i < s.length(); i++) {
            charCodeS += (int)(s.charAt(i));
        }
        
        for(int i = 0; i < t.length(); i++) {
            charCodeT += (int)(t.charAt(i));
        }
        
        return (char)(charCodeT - charCodeS);
    }
}
//You're given strings J representing the types of stones that are jewels, and S representing the stones you have.  
//Each character in S is a type of stone you have.  You want to know how many of the stones you have are also jewels.

//The letters in J are guaranteed distinct, and all characters in J and S are letters. Letters are case sensitive, 
//so "a" is considered a different type of stone from "A".

class JewelsAndStones {
    public int numJewelsInStones(String J, String S) {
        HashMap<Character, Integer> map = new HashMap<Character, Integer>();
        for(char c: J.toCharArray()) {
            map.put(c, 1);
        }
        
        int numberOfJewels = 0;
        for(char c: S.toCharArray()) {
            if(map.containsKey(c)) {
                numberOfJewels++;
            }
        }
        
        return numberOfJewels;
    }
}
// Given a string, we can "shift" each of its letter to its successive letter, for example: "abc" -> "bcd". We can keep "shifting" which forms the sequence:

// "abc" -> "bcd" -> ... -> "xyz"
// Given a list of strings which contains only lowercase alphabets, group all strings that belong to the same shifting sequence.

// For example, given: ["abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"], 
// A solution is:

// [
//   ["abc","bcd","xyz"],
//   ["az","ba"],
//   ["acef"],
//   ["a","z"]
// ]

public class GroupShiftedStrings {
    public List<List<String>> groupStrings(String[] strings) {
        List<List<String>> result = new ArrayList<List<String>>();
        
        HashMap<String, List<String>> map = new HashMap<String, List<String>>();
        
        for(String s : strings) {
            int offset = s.charAt(0) - 'a';
            String key = "";
            
            for(int i = 0; i < s.length(); i++) {
                char current = (char)(s.charAt(i) - offset);
                
                if(current < 'a') {
                    current += 26;
                }
                
                key += current;
            }
            
            if(!map.containsKey(key)) {
                List<String> list = new ArrayList<String>();
                map.put(key, list);
            }
            
            map.get(key).add(s);
        }
        
        for(String key : map.keySet()) {
            List<String> list = map.get(key);
            
            Collections.sort(list);
            
            result.add(list);   
        }
        
        return result;
    }
}
// Given two sparse matrices A and B, return the result of AB.

// You may assume that A's column number is equal to B's row number.

// Example:

// A = [
//   [ 1, 0, 0],
//   [-1, 0, 3]
// ]

// B = [
//   [ 7, 0, 0 ],
//   [ 0, 0, 0 ],
//   [ 0, 0, 1 ]
// ]


//      |  1 0 0 |   | 7 0 0 |   |  7 0 0 |
// AB = | -1 0 3 | x | 0 0 0 | = | -7 0 3 |
//                   | 0 0 1 |

public class SparseMatrixMultiplication {
    public int[][] multiply(int[][] A, int[][] B) {
        int m = A.length;
        int n = A[0].length;
        int nB = B[0].length;
        int[][] C = new int[m][nB];
        
        for(int i = 0; i < m; i++) {
            for(int j = 0; j < n; j++) {
                if(A[i][j] != 0) {
                    for(int k = 0; k < nB; k++) {
                        if(B[j][k] != 0) {
                            C[i][k] += A[i][j] * B[j][k];
                        }
                    }
                }
            }
        }
        
        return C;
    }
}
// Design a logger system that receive stream of messages along with its timestamps, each message should be printed if and only if it is not printed in the last 10 seconds.

// Given a message and a timestamp (in seconds granularity), return true if the message should be printed in the given timestamp, otherwise returns false.

// It is possible that several messages arrive roughly at the same time.

// Example:

// Logger logger = new Logger();

// // logging string "foo" at timestamp 1
// logger.shouldPrintMessage(1, "foo"); returns true; 

// // logging string "bar" at timestamp 2
// logger.shouldPrintMessage(2,"bar"); returns true;

// // logging string "foo" at timestamp 3
// logger.shouldPrintMessage(3,"foo"); returns false;

// // logging string "bar" at timestamp 8
// logger.shouldPrintMessage(8,"bar"); returns false;

// // logging string "foo" at timestamp 10
// logger.shouldPrintMessage(10,"foo"); returns false;

// // logging string "foo" at timestamp 11
// logger.shouldPrintMessage(11,"foo"); returns true;

public class LoggerRateLimiter {
    HashMap<String, Integer> messages;

    /** Initialize your data structure here. */
    public Logger() {
       this.messages = new HashMap<String, Integer>(); 
    }
    
    /** Returns true if the message should be printed in the given timestamp, otherwise returns false.
        If this method returns false, the message will not be printed.
        The timestamp is in seconds granularity. */
    public boolean shouldPrintMessage(int timestamp, String message) {
        if(messages.containsKey(message)) {
            if(timestamp - messages.get(message) >= 10) {
                messages.put(message, timestamp);
                return true;
            } else {
                return false;
            }
        } else {
            messages.put(message, timestamp);
            return true;
        }
    }
}

/**
 * Your Logger object will be instantiated and called as such:
 * Logger obj = new Logger();
 * boolean param_1 = obj.shouldPrintMessage(timestamp,message);
 */
// Given a string S and a string T, find the minimum window in S which will contain all the characters in T in complexity O(n).

// For example,
// S = "ADOBECODEBANC"
// T = "ABC"
// Minimum window is "BANC".

// Note:
// If there is no such window in S that covers all characters in T, return the empty string "".

// If there are multiple such windows, you are guaranteed that there will always be only one unique minimum window in S.

public class MinimumWindowSubstring {
    public String minWindow(String s, String t) {
        HashMap<Character, Integer> map = new HashMap<>();
        
        for(char c : s.toCharArray()) {
            map.put(c, 0);
        }
        
        for(char c : t.toCharArray()) {
            if(map.containsKey(c)) {
                map.put(c, map.get(c)+ 1);
            } else {
                return "";
            }
        }
        
        int start = 0;
        int end = 0;
        int minStart = 0;
        int minLength = Integer.MAX_VALUE;
        int counter = t.length();
        
        while(end < s.length()) {
            char c1 = s.charAt(end);
            
            if(map.get(c1) > 0) {
                counter--;
            }
            
            map.put(c1, map.get(c1) - 1);
            end++;
            
            while(counter == 0) {
                if(minLength > end - start) {
                    minLength = end - start;
                    minStart = start;
                }
                
                char c2 = s.charAt(start);
                map.put(c2, map.get(c2) + 1);
                
                if(map.get(c2) > 0) {
                    counter++;
                }
                
                start++;
            }
        }
        
        return minLength == Integer.MAX_VALUE ? "" : s.substring(minStart, minStart + minLength);
    }
}
//Given two lists Aand B, and B is an anagram of A. B is an anagram of A means B is made by randomizing the order of the elements in A.
//We want to find an index mapping P, from A to B. A mapping P[i] = j means the ith element in A appears in B at index j.
//These lists A and B may contain duplicates. If there are multiple answers, output any of them.

//For example, given
//A = [12, 28, 46, 32, 50]
//B = [50, 12, 32, 46, 28]

//We should return
//[1, 4, 3, 2, 0]
//as P[0] = 1 because the 0th element of A appears at B[1], and P[1] = 4 because the 1st element of A appears at B[4], and so on.

class FindAnagramMappings {
    public int[] anagramMappings(int[] A, int[] B) {
        int[] mapping = new int[A.length];
        HashMap<Integer, Integer> map = new HashMap<Integer, Integer>();
        
        for(int i = 0; i < B.length; i++) {
            map.put(B[i], i);
        }
        
        for(int i = 0; i < A.length; i++) {
            mapping[i] = map.get(A[i]);
        }
        
        return mapping;
    }
}
//You are playing the following Bulls and Cows game with your friend: You write down a number and ask your friend to guess what the number is. Each time your friend makes a guess, you provide a hint that indicates how many digits in said guess match your secret number exactly in both digit and position (called "bulls") and how many digits match the secret number but locate in the wrong position (called "cows"). Your friend will use successive guesses and hints to eventually derive the secret number.
//
//Write a function to return a hint according to the secret number and friend's guess, use A to indicate the bulls and B to indicate the cows. 
//
//Please note that both secret number and friend's guess may contain duplicate digits.
//
//Example 1:
//
//Input: secret = "1807", guess = "7810"
//
//Output: "1A3B"
//
//Explanation: 1 bull and 3 cows. The bull is 8, the cows are 0, 1 and 7.
//Example 2:
//
//Input: secret = "1123", guess = "0111"
//
//Output: "1A1B"
//
//Explanation: The 1st 1 in friend's guess is a bull, the 2nd or 3rd 1 is a cow.
//Note: You may assume that the secret number and your friend's guess only contain digits, and their lengths are always equal.

class BullsAndCows {
    public String getHint(String secret, String guess) {
        int bulls = 0;
        int cows = 0;
        int[] counts = new int[10];
        for(int i = 0; i < secret.length(); i++) {
            if(secret.charAt(i) == guess.charAt(i)) {
                bulls++;
            }  else {
                if(counts[secret.charAt(i) - '0']++ < 0) {
                    cows++;
                }
                if(counts[guess.charAt(i) - '0']-- > 0) {
                    cows++;
                }
            }
        }
        
        return bulls + "A" + cows + "B";
    }
}
// Given an array of strings, group anagrams together.

// For example, given: ["eat", "tea", "tan", "ate", "nat", "bat"], 
// Return:

// [
//   ["ate", "eat","tea"],
//   ["nat","tan"],
//   ["bat"]
// ]
// Note: All inputs will be in lower-case.

public class GroupAnagrams {
    public List<List<String>> groupAnagrams(String[] strs) {
        if(strs == null || strs.length == 0) {
            return new ArrayList<List<String>>();
        }
        
        HashMap<String, ArrayList<String>> map = new HashMap<String, ArrayList<String>>();
        
        Arrays.sort(strs);
        
        for(String s : strs) {
            char[] characters = s.toCharArray();
        
            Arrays.sort(characters);
            
            String key = String.valueOf(characters);
            
            if(!map.containsKey(key)) {
                map.put(key, new ArrayList<String>());
            }
            
            map.get(key).add(s);
        }
        
        return new ArrayList<List<String>>(map.values());
    }
}
//Determine if a Sudoku is valid, according to: Sudoku Puzzles - The Rules. (http://sudoku.com.au/TheRules.aspx)
//The Sudoku board could be partially filled, where empty cells are filled with the character '.'.
//A partially filled sudoku which is valid.

//Note:
//A valid Sudoku board (partially filled) is not necessarily solvable. Only the filled cells need to be validated.

class ValidSudoku {
    public boolean isValidSudoku(char[][] board) {
        for(int i = 0; i < board.length; i++){
            HashSet<Character> rows = new HashSet<Character>();
            HashSet<Character> columns = new HashSet<Character>();
            HashSet<Character> box = new HashSet<Character>();
            for (int j = 0; j < board[0].length; j++){
                if(board[i][j] != '.' && !rows.add(board[i][j])) {
                    return false;
                }
                if(board[j][i]!='.' && !columns.add(board[j][i])) {
                    return false;
                }
                int rowIndex = (i / 3) * 3;
                int columnIndex = (i % 3) * 3;
                if(board[rowIndex + j / 3][columnIndex + j % 3] != '.' && !box.add(board[rowIndex + j / 3][columnIndex + j % 3])) {
                    return false;
                }
            }
        }
        return true;
    }
}

// Given a binary tree, return the vertical order traversal of its nodes' values. (ie, from top to bottom, column by column).

// If two nodes are in the same row and column, the order should be from left to right.

// Examples:

// Given binary tree [3,9,20,null,null,15,7],
//    3
//   /\
//  /  \
//  9  20
//     /\
//    /  \
//   15   7
// return its vertical order traversal as:
// [
//   [9],
//   [3,15],
//   [20],
//   [7]
// ]
// Given binary tree [3,9,8,4,0,1,7],
//      3
//     /\
//    /  \
//    9   8
//   /\  /\
//  /  \/  \
//  4  01   7
// return its vertical order traversal as:
// [
//   [4],
//   [9],
//   [3,0,1],
//   [8],
//   [7]
// ]
// Given binary tree [3,9,8,4,0,1,7,null,null,null,2,5] (0's right child is 2 and 1's left child is 5),
//      3
//     /\
//    /  \
//    9   8
//   /\  /\
//  /  \/  \
//  4  01   7
//     /\
//    /  \
//    5   2
// return its vertical order traversal as:
// [
//   [4],
//   [9,5],
//   [3,0,1],
//   [8,2],
//   [7]
// ]

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class BinaryTreeVerticalOrderTraversal {
    public List<List<Integer>> verticalOrder(TreeNode root) {
        List<List<Integer>> result = new ArrayList<>();

        if(root == null) {
            return result;
        }
        
        Map<Integer, ArrayList<Integer>> map = new HashMap<>();
        Queue<TreeNode> q = new LinkedList<>();
        Queue<Integer> cols = new LinkedList<>();
        
        q.add(root);
        cols.add(0);
        
        int min = 0;
        int max = 0;
        
        while(!q.isEmpty()) {
            TreeNode node = q.poll();
            int col = cols.poll();
            
            if(!map.containsKey(col)) {
                map.put(col, new ArrayList<Integer>());
            }
            
            map.get(col).add(node.val);
            
            if(node.left != null) {
                q.add(node.left);
                cols.add(col - 1);
                min = Math.min(min, col - 1);
            }
            
            if(node.right != null) {
                q.add(node.right);
                cols.add(col + 1);
                max = Math.max(max, col + 1);
            }
        }
        
        for(int i = min; i <= max; i++) {
            result.add(map.get(i));
        }
        
        return result;
    }
}
// Given an array of integers, return indices of the two numbers such that they add up to a specific target.

// You may assume that each input would have exactly one solution, and you may not use the same element twice.

// Example:
// Given nums = [2, 7, 11, 15], target = 9,

// Because nums[0] + nums[1] = 2 + 7 = 9,
// return [0, 1].

public class TwoSum {
    public int[] twoSum(int[] nums, int target) {
        int[] result = new int[2];
        
        HashMap<Integer, Integer> map = new HashMap<>();
        
        for(int i = 0; i < nums.length; i++) {
            if(map.containsKey(target - nums[i])) {
                result[1] = i;
                result[0] = map.get(target - nums[i]);

                return result;
            }
            
            map.put(nums[i], i);
        }
        
        return result;
    }
}
// Implement int sqrt(int x).

// Compute and return the square root of x.

public class Solution {
    public int mySqrt(int x) {
        if(x == 0) {
            return 0;
        }
        
        int left = 1;
        int right = x;
        
        while(left <= right) {
            int mid = left + (right - left) / 2;
            
            if(mid == x / mid) {
                return mid;
            } else if(mid > x / mid) {
                right = mid - 1;
            } else if(mid < x / mid) {
                left = mid + 1;
            }
        }
        
        return right;
    }
}
// You are a product manager and currently leading a team to develop a new product. Unfortunately, the latest version of your product fails the quality check. Since each version is developed based on the previous version, all the versions after a bad version are also bad.

// Suppose you have n versions [1, 2, ..., n] and you want to find out the first bad one, which causes all the following ones to be bad.

// You are given an API bool isBadVersion(version) which will return whether version is bad. Implement a function to find the first bad version. You should minimize the number of calls to the API.

/* The isBadVersion API is defined in the parent class VersionControl.
      boolean isBadVersion(int version); */

public class FirstBadVersion extends VersionControl {
    public int firstBadVersion(int n) {
        int start = 1;
        int end = n;
        
        while(start < end) {
            int mid = start + (end - start) / 2;

            if(!isBadVersion(mid)) {
                start = mid + 1;
            } else {
                end = mid;
            }
        }
        
        return start;
    }
}
// Implement pow(x, n).

public class PowerOfXToTheN {
    public double myPow(double x, int n) {
        if(n == 0) {
            return 1;
        }
        
        if(Double.isInfinite(x)) {
            return 0;
        }
        
        if(n < 0) {
            n = -n;
            x = 1 / x;
        }
        
        return n % 2 == 0 ? myPow(x * x, n / 2) : x * myPow(x * x, n / 2);
    }
}
// Given a non-empty binary search tree and a target value, find the value in the BST that is closest to the target.

// Note:
	// Given target value is a floating point.
	// You are guaranteed to have only one unique value in the BST that is closest to the target.

/**
 * Definition for a binary tree node.
 * public class TreeNode {
 *     int val;
 *     TreeNode left;
 *     TreeNode right;
 *     TreeNode(int x) { val = x; }
 * }
 */
public class ClosestBinarySearchTreeValue {
    public int closestValue(TreeNode root, double target) {
        int value = root.val;
        TreeNode child = root.val < target ? root.right : root.left;

        if(child == null) {
            return value;
        }

        int childValue = closestValue(child, target);
        
        return Math.abs(value - target) < Math.abs(childValue - target) ? value : childValue;
    }
}
// We are playing the Guess Game. The game is as follows:

// I pick a number from 1 to n. You have to guess which number I picked.

// Every time you guess wrong, I'll tell you whether the number is higher or lower.

// You call a pre-defined API guess(int num) which returns 3 possible results (-1, 1, or 0):

// -1 : My number is lower
//  1 : My number is higher
//  0 : Congrats! You got it!
// Example:
// n = 10, I pick 6.

// Return 6.

public class GuessNumberHigherOrLower extends GuessGame {
    public int guessNumber(int n) {
        int left = 1;
        int right = n;
        while(left <= right) {
            int mid = left + (right - left) / 2;
            if(guess(mid) == 0) {
                return mid;
            } else if(guess(mid) > 0) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        return -1;
    }
}
