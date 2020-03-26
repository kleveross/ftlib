package main

import (
    /*
       #include <stdlib.h>
       typedef struct member_list{
           char *addrs[1024];
           int size;
       } member_list;
    */
    "C"
    "fmt"
    "github.com/hashicorp/memberlist"
    "os"
    "time"
)

var list *memberlist.Memberlist


func main() {
}

//export init_memberlist
func init_memberlist(
    cLogFileName *C.char,
    cBindAddr *C.char,
    cAdvertiseAddr *C.char,
    cTCPTimeout C.int,
    ) C.int {
    var err error
    config := memberlist.DefaultLocalConfig()

    logFileName := C.GoString(cLogFileName)
    customBindAddr := C.GoString(cBindAddr)
    customAdvertiseAddr := C.GoString(cAdvertiseAddr)
    customTCPTimeout := time.Duration(int64(cTCPTimeout))

    if customBindAddr != "" {
        fmt.Printf("Setting Bind Address as %s\n", customBindAddr)
        config.BindAddr = customBindAddr
    }

    if customAdvertiseAddr != "" {
        fmt.Printf("Setting Advertise Address as %s\n", customAdvertiseAddr)
        config.AdvertiseAddr = customAdvertiseAddr
    }

    if int64(cTCPTimeout) > 0 {
        fmt.Printf("Setting TCPTimeout as %d\n", int64(cTCPTimeout))
        config.TCPTimeout = customTCPTimeout * time.Second
    }

    if logFileName != "" {
        fmt.Printf("log file: %s\n", logFileName)
        f, err := os.Create(logFileName)
        if err != nil {
            fmt.Println(err)
            panic("Error! cannot create log file")
        }
        config.LogOutput = f
    }
    list, err = memberlist.Create(config)
    if err != nil {
        panic("Failed to create memberlist: " + err.Error())
        return 1
    }
    return 0
}

//export join
func join(ns []*C.char) C.int {
    fmt.Println("joining with ips:")
    addrList := make([]string, len(ns))
    for _, n := range ns {
        fmt.Printf("%s\n", C.GoString(n))
        addrList = append(addrList, C.GoString(n))
    }
    n, err := list.Join(addrList)
    if err != nil {
        fmt.Print("Failed to join cluster: " + err.Error())
        return 0
    }
    return C.int(n)
}

//export get_memberlist
func get_memberlist() C.member_list {
    // fill with the contemporary member list
    ml := C.member_list{}

    var size int
    for idx, member := range list.Members() {
        // fmt.Printf("Member: %s %s\n", member.Name, member.Addr)
        ml.addrs[idx] = C.CString(member.Addr.String())
        size = idx + 1
    }

    ml.size = C.int(size)

    return ml
}
