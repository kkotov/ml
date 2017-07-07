#ifndef csvUtils_h
#define csvUtils_h

#include <tuple>
#include <vector>
#include <string>
#include <unordered_map>

template<int IDX, int NMAX, typename... Args>
struct READ_TUPLE {
    static bool read(std::istream &in, std::tuple<Args...> &t){
        if( in.eof() ) return false;
        in >> std::get<IDX>(t);
        return READ_TUPLE<IDX+1,NMAX,Args...>::read(in,t);
    }
};

template<int NMAX, typename... Args>
struct READ_TUPLE<NMAX,NMAX,Args...>{
    static bool read(std::istream &in, std::tuple<Args...> &t){ return true; }
};

template <typename... Args>
bool read_tuple(std::istream &in, std::tuple<Args...> &t) noexcept {
    return READ_TUPLE<0,sizeof...(Args),Args...>::read(in,t);
}

void setCommaDelim(std::istream& input){
    struct field_reader: std::ctype<char> {
        field_reader(): std::ctype<char>(get_table()) {}

        static std::ctype_base::mask const* get_table() {
            static std::vector<std::ctype_base::mask> 
                rc(table_size, std::ctype_base::mask());

            rc['\n'] = std::ctype_base::space;
            rc[':']  = std::ctype_base::space;
            rc[',']  = std::ctype_base::space;
            return &rc[0];
        }
    };
    input.imbue(std::locale(std::locale(), new field_reader()));
}

#endif
