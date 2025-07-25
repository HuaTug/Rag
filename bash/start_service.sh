#!/bin/bash

# 智能RAG服务启动脚本

set -e

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 日志函数
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

log_debug() {
    echo -e "${BLUE}[DEBUG]${NC} $1"
}

# 检查Python环境
check_python() {
    log_info "检查Python环境..."
    
    if ! command -v python3 &> /dev/null; then
        log_error "Python3 未安装"
        exit 1
    fi
    
    python_version=$(python3 --version | cut -d' ' -f2)
    log_info "Python版本: $python_version"
    
    # 检查版本是否 >= 3.8
    if ! python3 -c "import sys; assert sys.version_info >= (3, 8)"; then
        log_error "需要Python 3.8或更高版本"
        exit 1
    fi
}

# 安装依赖
install_dependencies() {
    log_info "安装Python依赖..."
    
    if [ ! -f "requirements.txt" ]; then
        log_error "requirements.txt 文件不存在"
        exit 1
    fi
    
    # 创建虚拟环境（如果不存在）
    if [ ! -d "venv" ]; then
        log_info "创建虚拟环境..."
        python3 -m venv venv
    fi
    
    # 激活虚拟环境
    source venv/bin/activate
    
    # 升级pip
    pip install --upgrade pip
    
    # 安装依赖
    pip install -r requirements.txt
    
    log_info "依赖安装完成"
}

# 检查配置文件
check_config() {
    log_info "检查配置文件..."
    
    if [ ! -f "service_config.json" ]; then
        log_warn "service_config.json 不存在，将创建默认配置"
        cp service_config.json.example service_config.json 2>/dev/null || true
    fi
    
    # 检查必要的环境变量
    if [ -z "$LLM_API_KEY" ]; then
        log_warn "LLM_API_KEY 环境变量未设置"
    fi
    
    log_info "配置检查完成"
}

# 创建必要目录
create_directories() {
    log_info "创建必要目录..."
    
    mkdir -p logs
    mkdir -p data
    mkdir -p temp
    
    log_info "目录创建完成"
}

# 运行测试
run_tests() {
    log_info "运行基础测试..."
    
    source venv/bin/activate
    
    # 运行智能分析器测试
    python test_intelligent_rag.py
    
    log_info "测试完成"
}

# 启动服务
start_service() {
    local mode=${1:-dev}
    
    log_info "启动RAG服务 (模式: $mode)..."
    
    source venv/bin/activate
    
    case $mode in
        "dev")
            # 开发模式
            uvicorn rag_service:app --host 0.0.0.0 --port 8000 --reload --log-level info
            ;;
        "prod")
            # 生产模式
            uvicorn rag_service:app --host 0.0.0.0 --port 8000 --workers 4 --log-level warning
            ;;
        "docker")
            # Docker模式
            docker-compose up --build
            ;;
        *)
            log_error "未知的启动模式: $mode"
            log_info "可用模式: dev, prod, docker"
            exit 1
            ;;
    esac
}

# 停止服务
stop_service() {
    log_info "停止RAG服务..."
    
    # 停止Docker服务
    if command -v docker-compose &> /dev/null; then
        docker-compose down
    fi
    
    # 停止本地服务
    pkill -f "uvicorn rag_service:app" || true
    
    log_info "服务已停止"
}

# 显示帮助信息
show_help() {
    echo "智能RAG服务管理脚本"
    echo ""
    echo "用法: $0 [命令] [选项]"
    echo ""
    echo "命令:"
    echo "  install     - 安装依赖和设置环境"
    echo "  test       - 运行测试"
    echo "  start      - 启动服务 [dev|prod|docker]"
    echo "  stop       - 停止服务"
    echo "  restart    - 重启服务"
    echo "  status     - 检查服务状态"
    echo "  logs       - 查看日志"
    echo "  help       - 显示帮助信息"
    echo ""
    echo "示例:"
    echo "  $0 install              # 安装依赖"
    echo "  $0 start dev            # 开发模式启动"
    echo "  $0 start prod           # 生产模式启动"
    echo "  $0 start docker         # Docker模式启动"
    echo ""
}

# 检查服务状态
check_status() {
    log_info "检查服务状态..."
    
    # 检查本地服务
    if pgrep -f "uvicorn rag_service:app" > /dev/null; then
        log_info "本地服务: 运行中"
    else
        log_warn "本地服务: 未运行"
    fi
    
    # 检查Docker服务
    if command -v docker-compose &> /dev/null; then
        if docker-compose ps | grep "Up" > /dev/null; then
            log_info "Docker服务: 运行中"
        else
            log_warn "Docker服务: 未运行"
        fi
    fi
    
    # 测试API
    if curl -s http://localhost:8000/health > /dev/null; then
        log_info "API健康检查: 通过"
    else
        log_warn "API健康检查: 失败"
    fi
}

# 查看日志
view_logs() {
    local lines=${1:-100}
    
    log_info "查看最近 $lines 行日志..."
    
    if [ -f "logs/rag_service.log" ]; then
        tail -n $lines logs/rag_service.log
    elif [ -f "rag_service.log" ]; then
        tail -n $lines rag_service.log
    else
        log_warn "未找到日志文件"
    fi
}

# 主函数
main() {
    case ${1:-help} in
        "install")
            check_python
            install_dependencies
            check_config
            create_directories
            ;;
        "test")
            run_tests
            ;;
        "start")
            start_service ${2:-dev}
            ;;
        "stop")
            stop_service
            ;;
        "restart")
            stop_service
            sleep 2
            start_service ${2:-dev}
            ;;
        "status")
            check_status
            ;;
        "logs")
            view_logs ${2:-100}
            ;;
        "help"|"--help"|"-h")
            show_help
            ;;
        *)
            log_error "未知命令: $1"
            show_help
            exit 1
            ;;
    esac
}

# 执行主函数
main "$@"
