

# Git问题
若开了外网，需要如下设置
1. 使用git config --global --unset http.proxy 删除配置；
2. 打开[ip查询](http://ipaddress.com) 查询下面两个域名，并记录IP(因为你开外网了，所以IP会经常改变)，然后修改Host；
3. 140.82.113.3 github.com  
4. 199.232.69.194 github.global.ssl.fastly.net
5. 最后Push

## OpenSSL SSL_read: Connection was reset, errno 10054
解除ssl验证
git config --global http.sslVerify "false"

## Failed to connect to github.com port 443: Timed out
取消全局代理：
git config --global --unset http.proxy
git config --global --unset https.proxy


