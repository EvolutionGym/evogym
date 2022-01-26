#include "BBTreeNode.h"



BBTreeNode::BBTreeNode()
{
}

BBTreeNode::BBTreeNode(int a_index, int b_index) {
	BBTreeNode::a_index = a_index;
	BBTreeNode::b_index = b_index;
	BBTreeNode::is_leaf = false;
	BBTreeNode::is_finished = false;
}

BBTreeNode::BBTreeNode(int boxel_index) {
	BBTreeNode::boxel_index = boxel_index;
	BBTreeNode::is_leaf = true;
	BBTreeNode::is_finished = false;
}

BBTreeNode::~BBTreeNode()
{
}
