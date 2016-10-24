// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef app_h
#define app_h

#include <string>
#include <vector>
#include <deque>
#include <list>
#include <set>
#include <map>
#include <utility> //for pair
#include <ostream>
#include <sstream>
#include "matrix.h"
#include "vec.h"


// This file provides "to_str(" functions that will convert just about anything to a std::string.


/// Convert any type that has a stream-insertion operator << to a string
template<typename T>
std::string to_str(const T& n)
{
	std::ostringstream os;
	os.precision(14);
	os << n;
	return os.str();
}

/// Convert a Vec to a string
extern std::string to_str(const Vec& v);

/// Convert a Matrix to a string
extern std::string to_str(const Matrix& v);

/// Convert any collection with a standard iterator to a string
template<typename T>
std::string to_str(T begin, T end, std::string spec = "")
{
	std::ostringstream os;
	os.precision(14);
	os << "[" << spec; 
	if(spec != ""){
	  os << ":";
	} 
	while(begin != end){ 
	  os << to_str(*begin); ++begin; 
	  if(begin != end){ os << ","; }
	}
	os << "]";
	return os.str();
}

/// Convert a vector to a string
template<typename T>
std::string to_str(const std::vector<T>& v){
	return to_str(v.begin(), v.end(),"vector");
}

/// Convert a list to a string
template<typename T>
std::string to_str(const std::list<T>& v){
	return to_str(v.begin(), v.end(),"list");
}

/// Convert a set to a string
template<typename T>
std::string to_str(const std::set<T>& v){
	return to_str(v.begin(), v.end(),"set");
}

/// Convert a deque to a string
template<typename T>
std::string to_str(const std::deque<T>& v){
	return to_str(v.begin(), v.end(),"deque");
}

/// Convert a multiset to a string
template<typename T>
std::string to_str(const std::multiset<T>& v){
	return to_str(v.begin(), v.end(),"multiset");
}

/// Convert a multimap to a string
template<typename Key, typename T>
std::string to_str(const std::multimap<Key, T>& v){
	return to_str(v.begin(), v.end(),"multimap");
}

/// Convert a map to a string
template<typename Key, typename T>
std::string to_str(const std::map<Key, T>& v){
	return to_str(v.begin(), v.end(),"map");
}


#endif // app_h
