"""
二叉树（Binary tree）是树形结构的一个重要类型。本项目以二叉树为例练习Python中的类与对象相关知识。

Author: <NAME>

DATE: 2021-05-11
"""


class Node:
    """
    二叉树节点类
    """
    def __init__(self, value: int):
        """
        初始化节点
        :param value: 节点值
        """
        self.value = value
        self.left = None
        self.right = None


class BinaryTree:
    """
    二叉树类
    """
    def __init__(self):
        """
        初始化二叉树
        """
        self.root = None

    def add_node(self, value: int):
        """
        向二叉树中添加节点
        :param value: 节点值
        """
        pass

    def _add_node(self, value: int, node: Node):
        """
        向二叉树中添加节点的递归函数
        :param value: 节点值
        :param node: 父节点
        """
        pass

    def search_node(self, value: int):
        """
        在二叉树中搜索节点
        :param value: 节点值
        :return: 节点对象
        """
        pass

    def _search_node(self, value: int, node: Node):
        """
        在二叉树中搜索节点的递归函数
        :param value: 节点值
        :param node: 父节点
        :return: 节点对象
        """
        pass

    def delete_node(self, value: int):
        """
        删除二叉树中的节点
        :param value: 节点值
        """
        pass

    def _delete_node(self, value: int, node: Node):
        """
        删除二叉树中的节点的递归函数
        :param value: 节点值
        :param node: 父节点
        """
        pass

    def inorder_traversal(self) -> list:
        """
        中序遍历二叉树
        :return: 节点值列表
        """
        pass

    def _inorder_traversal(self, node: Node) -> list:
        """
        中序遍历二叉树的递归函数
        :param node: 父节点
        """
        pass

    def preorder_traversal(self) -> list:
        """
        先序遍历二叉树
        """
        pass

    def _preorder_traversal(self, node: Node) -> list:
        """
        先序遍历二叉树的递归函数
        :param node: 父节点
        """
        pass

    def postorder_traversal(self) -> list:
        """
        后序遍历二叉树
        """
        pass

    def _postorder_traversal(self, node: Node) -> list:
        """
        后序遍历二叉树的递归函数
        :param node: 父节点
        """
        pass

    def height(self) -> int:
        """
        计算二叉树的高度
        :return: 高度
        """
        pass

    def _height(self, node: Node) -> int:
        """
        计算二叉树的高度的递归函数
        :param node: 父节点
        :return: 高度
        """
        pass

    def is_balanced(self) -> bool:
        """
        判断二叉树是否平衡
        :return: 布尔值
        """
        pass

    def _is_balanced(self, node: Node) -> bool:
        """
        判断二叉树是否平衡的递归函数
        :param node: 父节点
        :return: 布尔值
        """
        pass

    def is_full(self) -> bool:
        """
        判断二叉树是否为满二叉树
        :return: 布尔值
        """
        pass

    def _is_full(self, node: Node) -> bool:
        """
        判断二叉树是否为满二叉树的递归函数
        :param node: 父节点
        :return: 布尔值
        """
        pass

    def is_complete(self) -> bool:
        """
        判断二叉树是否为完全二叉树
        :return: 布尔值
        """
        pass

    def _is_complete(self, node: Node) -> bool:
        """
        判断二叉树是否为完全二叉树的递归函数
        :param node: 父节点
        :return: 布尔值
        """
        pass

    def __str__(self) -> str:
        """
        打印二叉树
        :return: 字符串
        """
        pass

    def _str(self, node: Node, level: int) -> str:
        """
        打印二叉树的递归函数
        :param node: 父节点
        :param level: 层级
        :return: 字符串
        """
        pass


# 测试代码，请勿修改
if __name__ == '__main__':
    # 创建二叉树
    tree = BinaryTree()
    # 向二叉树中添加节点
    tree.add_node(3)
    tree.add_node(9)
    tree.add_node(4)
    tree.add_node(7)
    tree.add_node(1)
    tree.add_node(2)
    tree.add_node(6)
    tree.add_node(8)
    tree.add_node(5)
    # 打印二叉树
    print(tree)
    # 中序遍历二叉树
    tree.inorder_traversal()
    # 先序遍历二叉树
    tree.preorder_traversal()
    # 后序遍历二叉树
    tree.postorder_traversal()
    # 计算二叉树的高度
    print(tree.height())
    # 判断二叉树是否平衡
    print(tree.is_balanced())
    # 判断二叉树是否为满二叉树
    print(tree.is_full())
    # 判断二叉树是否为完全二叉树
    print(tree.is_complete())
    # 删除二叉树中的节点
    tree.delete_node(9)
    # 打印二叉树
    print(tree)

