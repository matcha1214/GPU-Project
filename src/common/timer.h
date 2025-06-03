#ifndef COMMON_TIMER_H
#define COMMON_TIMER_H

#include <chrono>
#include <string>
#include <map>
#include <iostream>

class Timer {
public:
    Timer();
    
    // Start timing a named section
    void start(const std::string& name);
    
    // Stop timing a named section and return elapsed time in milliseconds
    double stop(const std::string& name);
    
    // Get elapsed time for a named section without stopping
    double elapsed(const std::string& name) const;
    
    // Reset all timers
    void reset();
    
    // Print timing summary
    void printSummary() const;
    
    // Get total time for a named section (accumulated across multiple start/stop cycles)
    double getTotalTime(const std::string& name) const;
    
    // Get call count for a named section
    int getCallCount(const std::string& name) const;

private:
    using TimePoint = std::chrono::high_resolution_clock::time_point;
    
    struct TimerData {
        TimePoint start_time;
        double total_time = 0.0;
        int call_count = 0;
        bool is_running = false;
    };
    
    std::map<std::string, TimerData> timers_;
    
    // Helper function to get current time
    static TimePoint now();
    
    // Helper function to calculate duration in milliseconds
    static double duration_ms(const TimePoint& start, const TimePoint& end);
};

// RAII timer for automatic timing of scopes
class ScopedTimer {
public:
    ScopedTimer(Timer& timer, const std::string& name);
    ~ScopedTimer();
    
private:
    Timer& timer_;
    std::string name_;
};

// Macro for easy scoped timing
#define SCOPED_TIMER(timer, name) ScopedTimer _scoped_timer(timer, name)

// Global timer instance
extern Timer g_timer;

#endif // COMMON_TIMER_H
