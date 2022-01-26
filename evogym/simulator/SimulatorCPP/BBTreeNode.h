#ifndef BB_TREE_NODE_H
#define BB_TREE_NODE_H

#include "main.h"
#include <vector>
#include <unordered_set>

class BBTreeNode
{
public:

	bool is_leaf;
	bool is_finished;
	
	int a_index;
	int b_index;
	int boxel_index;

	BoundingBox bbox;

	unordered_set<int> neighbors;

	BBTreeNode();
	BBTreeNode(int a_index, int b_index);
	BBTreeNode(int boxel_index);
	~BBTreeNode();
};


#endif // !BB_TREE_NODE_H
