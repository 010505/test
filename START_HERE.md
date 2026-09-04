# GestureGraph Lab 交接入口

这是可直接转交组员的轻量版本。先阅读 `handoff/GestureGraph_Lab_Handoff.docx`，明天介绍项目时打开 `handoff/TOMORROW_BRIEFING.md`。

## 一键启动

- macOS：双击 `handoff/start_mac.command`。若系统拦截，右键文件选择“打开”。
- Windows 10/11：双击 `handoff/start_windows.bat`。
- 首次启动会创建 Python 虚拟环境、安装依赖，并在浏览器打开前启动本地服务；完成后访问 `http://localhost:8080/`。

建议环境：64 位 Python 3.10–3.12、Node.js 20+、带摄像头的 Edge/Chrome/Safari。首次安装需要联网。

## 当前可信结论

- macOS：已在 Apple Silicon Mac 上完成环境检查、前后端测试、模型加载和摄像头页面运行验证。
- Windows：环境检查、前后端测试、模型接口、页面资源和真实摄像头权限均已完成验证。
- 官方 SHREC'17 测试集上的 ST-GCN 准确率为 70.95%；摄像头实时场景属于额外域外演示，不能直接把该数值当作现场准确率。

## 交接包边界

本包包含运行网页、14 类模型、MediaPipe 手部检测模型、14 个参考动画、测试、实验报告、第一版展示 PPT 和交接文档。为便于传输并保护原采集数据，本包不包含 `.venv`、`node_modules`、约 6 GB 的完整 SHREC'17 数据集、个人录制样本或早期 smoke 数据。

## 接手后的第一件事

组员接手后先执行对应系统的启动脚本，再确认网页、摄像头、手部骨架、Model Space 与 14 类模型输出正常；课程汇报可直接从 `handoff/GestureGraph_Lab_Presentation_v1.pptx` 开始修改。
