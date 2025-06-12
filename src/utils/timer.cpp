#include "timer.h"
#include <iostream>
#include <iomanip>

// Global timer instance
Timer g_timer;

Timer::Timer() {
}

void Timer::start(const std::string& name) {
    TimerData& data = timers_[name];
    if (data.is_running) {
        std::cerr << "Warning: Timer '" << name << "' is already running!" << std::endl;
        return;
    }
    
    data.start_time = now();
    data.is_running = true;
}

double Timer::stop(const std::string& name) {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        std::cerr << "Warning: Timer '" << name << "' was never started!" << std::endl;
        return 0.0;
    }
    
    TimerData& data = it->second;
    if (!data.is_running) {
        std::cerr << "Warning: Timer '" << name << "' is not running!" << std::endl;
        return 0.0;
    }
    
    TimePoint end_time = now();
    double elapsed_ms = duration_ms(data.start_time, end_time);
    
    data.total_time += elapsed_ms;
    data.call_count++;
    data.is_running = false;
    
    return elapsed_ms;
}

double Timer::elapsed(const std::string& name) const {
    auto it = timers_.find(name);
    if (it == timers_.end()) {
        return 0.0;
    }
    
    const TimerData& data = it->second;
    if (!data.is_running) {
        return 0.0;
    }
    
    TimePoint current_time = now();
    return duration_ms(data.start_time, current_time);
}

void Timer::reset() {
    timers_.clear();
}

void Timer::printSummary() const {
    if (timers_.empty()) {
        std::cout << "No timing data available." << std::endl;
        return;
    }
    
    std::cout << "\n=== Timer Summary ===" << std::endl;
    std::cout << std::left << std::setw(20) << "Timer Name" 
              << std::setw(15) << "Total Time (ms)"
              << std::setw(10) << "Calls"
              << std::setw(15) << "Avg Time (ms)" << std::endl;
    std::cout << std::string(60, '-') << std::endl;
    
    for (const auto& [name, data] : timers_) {
        double avg_time = data.call_count > 0 ? data.total_time / data.call_count : 0.0;
        std::cout << std::left << std::setw(20) << name
                  << std::setw(15) << std::fixed << std::setprecision(2) << data.total_time
                  << std::setw(10) << data.call_count
                  << std::setw(15) << std::fixed << std::setprecision(2) << avg_time
                  << std::endl;
    }
    std::cout << "=====================" << std::endl;
}

double Timer::getTotalTime(const std::string& name) const {
    auto it = timers_.find(name);
    return (it != timers_.end()) ? it->second.total_time : 0.0;
}

int Timer::getCallCount(const std::string& name) const {
    auto it = timers_.find(name);
    return (it != timers_.end()) ? it->second.call_count : 0;
}

Timer::TimePoint Timer::now() {
    return std::chrono::high_resolution_clock::now();
}

double Timer::duration_ms(const TimePoint& start, const TimePoint& end) {
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    return duration.count() / 1000.0; // Convert to milliseconds
}

// ScopedTimer implementation
ScopedTimer::ScopedTimer(Timer& timer, const std::string& name) 
    : timer_(timer), name_(name) {
    timer_.start(name_);
}

ScopedTimer::~ScopedTimer() {
    timer_.stop(name_);
} 