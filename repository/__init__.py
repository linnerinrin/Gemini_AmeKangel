#"""
#架构作用：repository包标记；本文件无导入/执行语句，保持包导入本身轻量
#调用方通过repository.conn获取db，通过repository.init_db显式初始化结构
#数据流发生在上述子模块，本文件不创建连接、不访问磁盘或执行数据库请求
#"""
